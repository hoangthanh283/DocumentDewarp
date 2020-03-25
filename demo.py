#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "ThanhHoang <hoangducthanh283@gmail.com>"

import os
import sys
import random
import logging
import numpy as np

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.spatial import distance as dist

from models.keypoint_net import KeyPointNet
from utils.keypoints import (
    extract_keypoints, group_keypoints
)
from utils.load_state import load_state
from utils.pose import Pose, propagate_ids
from tools import (
    normalize,
    pad_width,
    four_point_transform
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DewarpModel(object):
    """Inference for dewarp for document camera images. 
    Args:
        weights_path (str): path to model weight 
    """
    def __init__(self, weights_path=None):
        # check device
        self.device = torch.device('cuda' \
            if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.ToTensor()

        # load model
        self.model, self.prefered_size = \
            self.load_model(weights_path)
        self.model.eval()

        # preprocess image prarams
        self.bordersize = 111
        self.stride = 8
        self.upsample_ratio = 4
        self.pad_value = (0, 0, 0)
        self.img_mean = (128, 128, 128)
        self.img_scale= 1 / 256
        self.num_keypoints = Pose.num_kpts

    def load_model(self, weights_path):
        """ Load pytorch model
        Args:
            net (torch.Model): input net graph
            weights_path (str): checkpoint model path
        Return:
            net (torch.Model): model loaded weights
        """

        # load checkpoint and get model configs
        checkpoint = torch.load(weights_path, map_location=self.device)
        prefered_size = checkpoint.get("prefered_size", 0)
        state_dict = checkpoint.get("state_dict")

        # load model dict
        model = KeyPointNet()
        load_state(model, state_dict)
        return (model.to(self.device), prefered_size)

    def preprocess(self, img):
        """ Preprocess input image
        Args:
            img (np.array): input image expected with grayscale
        Return:
            tensor_img (torch.tensor): tensor image
        """

        # check and convert gray scale image to rgb
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if img.shape[0] > img.shape[1]:
            top_pad = self.bordersize
            lef_pad = (self.bordersize * 2 + img.shape[0] - img.shape[1]) // 2
        else:
            top_pad = (self.bordersize * 2 + img.shape[1] - img.shape[0]) // 2
            lef_pad = self.bordersize

        # padding images
        padding_warped_image = cv2.copyMakeBorder(img, \
            top=top_pad,
            bottom=top_pad,
            left=lef_pad,
            right=lef_pad,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0])

        if padding_warped_image.size == 0:
            raise IOError('Image {} cannot be read')

        # resize and normalize images with prefered size and norm values
        height, width = padding_warped_image.shape[:2]
        scale = self.prefered_size / height

        scaled_img = cv2.resize(padding_warped_image, (0, 0), \
            fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, self.img_mean, self.img_scale)

        min_dims = [self.prefered_size, max(scaled_img.shape[1], self.prefered_size)]
        padded_img, pad = pad_width(scaled_img, self.stride, self.pad_value, min_dims)
        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        return (tensor_img.to(self.device), padding_warped_image, scale, pad)

    def validate_keypoints(self, pts, theta=0.3):
        """ Validate all predicted keypoints, only get 4 keypoints (corners).
        Args:
            pts (list): list of all keypoints positions [(x1, y1), (x2, y2), ..]
            theta (float): threshold to filter failure keypoints
        Return:
            True or False (Boolean): True is valid keypoints, else False
        """

        # Return fail as the number of keypoints is not equal to 4.
        if len(pts) != 4:
            return False
        assert theta < 1.0, "Theta must be smaller than 1 !"
        pts = np.asarray(pts)

        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # sort the left-most coordinates
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # calculate Euclidean distance between the
        # top-left and right-most points
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        new_pts = np.array([tl, tr, br, bl], dtype=np.int32)

        # get the size of all edges in the rect
        width_a = abs(new_pts[0][0] - new_pts[1][0])
        width_b = abs(new_pts[2][0] - new_pts[3][0])

        height_a = abs(new_pts[0][1] - new_pts[3][1])
        height_b = abs(new_pts[1][1] - new_pts[2][1])

        # if too skew rect, return False
        if (min(width_a, width_b) / max(width_a, width_b) <= theta):
            return False
        elif (min(height_a, height_b) / max(height_a, height_b) <= theta):
            return False
        else:
            return True

    def predict(self, img):
        """ predict skewed angle and re-rotate image
        Args:
            img (np.array): input image with expected grayscale
        Return:
            img (np.array): output rotated image with shape is same with input image
            angle (int): detected angle
        """

        # preprocess input image
        input_tensor, preprocess_img, scale, pad = self.preprocess(img)

        with torch.no_grad():
            input_var = torch.autograd.Variable(input_tensor)

            # Get prediction
            stages_output = self.model(input_var)

        # scale the heatmap
        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(\
            stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), \
            fx=self.upsample_ratio, fy=self.upsample_ratio, \
            interpolation=cv2.INTER_CUBIC)

        # scale the paf predictions
        stage2_pafs = stages_output[-1]
        pafs = np.transpose(\
            stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), \
            fx=self.upsample_ratio, fy=self.upsample_ratio, \
            interpolation=cv2.INTER_CUBIC)

        # extract keypoints from heatmaps
        all_keypoints_by_type = []
        total_keypoints_num = 0
        for kpt_idx in range(self.num_keypoints):
            total_keypoints_num += extract_keypoints(\
                heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        # convert heatmaps to keypoint locations
        pose_entries, all_keypoints = group_keypoints(\
            all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * \
                self.stride / self.upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * \
                self.stride / self.upsample_ratio - pad[0]) / scale

        # valid keypoints, only get valid 4-points, else return origin image
        n_keypoints = list(map(lambda k: \
            [int(k[0]), int(k[1]), k[2], int(k[3])], all_keypoints))
        pts = list(map(lambda p: [p[0], p[1]], n_keypoints))
        x, y, rect_w, rect_h = cv2.boundingRect(np.array(pts))
        is_valid = self.validate_keypoints(pts)

        if is_valid == True:
            valid_pts = pts
        else:
            if len(pts) >= 3:
                valid_pts = [[x, y], [x + rect_w, y], \
                    [x + rect_w, y + rect_h], [x, y + rect_h]]
            else:
                valid_pts = None

        if valid_pts:
            affine_trans_img = four_point_transform(preprocess_img, valid_pts)
        else:
            affine_trans_img = img
        return (affine_trans_img, len(pts))

    def process(self, images):
        """
        Rotate input images
        :param images: a single `str`, `pathlib.Path`, `numpy.ndarray`
                    or an iterable containing them
        :return: a list of dict containing information of rotated images
        if there is any error then the original input image is appended,
            with undefine confidence and angle value
        """
        if type(images) == list and not images:
            return []

        results = []
        for img in images:
            single_result = dict()
            try:
                dewarped_img, num_points = self.predict(img)
                logger.debug("Num points: {}, Confidence: {}".format(\
                    str(num_points), str(0.99)))

                single_result["output"] = dewarped_img
                single_result["confidence"] = 0.99
                single_result["metadata"] = {
                    "num_points": num_points
                }
                results.append(single_result)
            except Exception as e:
                logger.error("Error warping image: {}".format(str(e)))
                single_result["output"] = img
                single_result["confidence"] = None
                single_result["metadata"] = {
                    "num_points": None
                }
                results.append(single_result)
        return results


if __name__ == "__main__":
    """ Unit test """
    image_folder = ""
    model_path = ""
    list_files = list(map(lambda f: \
        os.path.join(image_folder, f), os.path.listdirs(image_folder)))

    model = DewarpModel(model_path)
    outputs = model.process(list_files)