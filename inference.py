#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "ThanhHoang <hoangducthanh283@gmail.com>"
__status__ = "Modules"


import os
import sys
import gc
import shutil
import random
import logging
import numpy as np

import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

from models.keypoint_net import KeyPointNet
from utils.keypoints import (
    extract_keypoints, group_keypoints
)
from utils.load_state import load_state
from utils.pose import Pose, propagate_ids
from utils.tools import (
    normalize,
    pad_width,
    four_point_transform
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class DewarpModel:
    """
    Inference for dewarp for document camera images.
    Model weight should be a dict, contain:
        state_dict (torch.modules): model weights
        prefered_size (int): size of image to normalize (Ex: 256)
    """
    def __init__(self, opt, weights_path=None):
        # check device
        self.device = torch.device('cuda' \
            if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.ToTensor()
        self.opt = opt

        # load model
        self.model, self.prefered_size = \
            self.load_model(weights_path)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.model = self.model.to(self.device)

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
        checkpoint = torch.load(\
            weights_path, map_location=self.device)
        prefered_size = checkpoint.get("prefered_size", 256)
        state_dict = checkpoint.get("state_dict")

        # load model dict
        model = KeyPointNet(self.opt)
        # load_state(model, state_dict)
        model.load_state_dict(state_dict)
        return (model, prefered_size)

    def preprocess(self, img):
        """ Preprocess input image
        Args:
            img (np.array): input image expected with grayscale
        Return:
            tensor_img (torch.tensor): tensor image
        """

        # check and convert gray scale image to rgb
        if isinstance(img, str) == True:
            img = cv2.imread(img, cv2.IMREAD_COLOR)
        elif not isinstance(img, np.array):
            assert("Currently only support for input str or numpy array!")

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

        meta_data = {
            "h_padding": top_pad,
            "w_padding": lef_pad,
            "scale": scale,
            "pad": pad
        }
        return (tensor_img.to(self.device), padding_warped_image, img, meta_data)

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
        with torch.no_grad():
            # preprocess input image
            input_tensor, preprocess_img, origin_img, meta = self.preprocess(img)

            # Get prediction
            stages_output = self.model(input_tensor)

            # get pad and scale of original image 
            scale = meta["scale"]
            pad = meta["pad"]

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
                affine_trans_img, affine_matrix = \
                    four_point_transform(preprocess_img, valid_pts)
            else:
                affine_trans_img = origin_img
                h, w = origin_img.shape[:2]
                valid_pts = [[0, 0], [w, 0], [w, h], [0, h]]
                affine_matrix= None
            meta["affine_matrix"] = affine_matrix
            meta["points"] = valid_pts
            return (affine_trans_img, meta)

    @staticmethod
    def map_point(point, meta={}):
        """ Transform original points to new point
            in warped coordinate system by using affine matrix.
        Esample: dst(x, y) = point_mapping(src(x, y), M)
        Ref: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective
        Args:
            point (list): a list of point position (x, y)
            meta (dict): meta data from output's self.process()
                affine_matrix (np.array): matrix to transform points
                h_padding (int): horizontal padding value in dewarped image
                w_padding (int): vertical padding value in dewarped image
        Return:
            point (list): a list of warped points (n_x, n_y)
        """
        # _, M = cv2.invert(M)
        assert len(point) == 2, "Input `point` should have size of 2, (x,y)"
        affine_matrix = meta.get("affine_matrix", None)
        h_padding = meta.get("h_padding", 0)
        w_padding = meta.get("w_padding", 0)

        # add padding as same as preprocess input image in dewarp model
        x, y = point
        x += w_padding
        y += h_padding
        if affine_matrix is None:
            return point

        # Calculate the new x', y' in warping coordinate system
        n_x = (affine_matrix[0][0]*x + affine_matrix[0][1]*y + affine_matrix[0][2]) \
            / (affine_matrix[2][0]*x + affine_matrix[2][1]*y + affine_matrix[2][2])
        n_y = (affine_matrix[1][0]*x + affine_matrix[1][1]*y + affine_matrix[1][2]) \
            / (affine_matrix[2][0]*x + affine_matrix[2][1]*y + affine_matrix[2][2])
        return (int(max(n_x, 0)), int(max(n_y, 0)))

    @classmethod
    def map_rect(cls, rect, meta={}):
        """ Transform original rect to new rect
            in warped coordinate system by using affine matrix.
        Esample: dst(x, y, w, h) = point_mapping(src(x, y, w, h), M)
        Ref: https://docs.opencv.org/2.4/modules/imgproc/doc/geometric_transformations.html#warpperspective
        Args:
            rect (list): a list of rectangle position (x, y, w, h)
            meta (dict): meta data from output's self.process()
                affine_matrix (np.array): matrix to transform points
                h_padding (int): horizontal padding value in dewarped image
                w_padding (int): vertical padding value in dewarped image
        Return:
            new_rect (list): a list of warped rect (n_x, n_y, n_w, n_h)
        """
        # _, M = cv2.invert(M)
        assert len(rect) == 4, "Input `rect` should have size of 4, (x,y,w,h)"
        affine_matrix = meta.get("affine_matrix", None)
        if affine_matrix is None:
            return rect

        x1, y1, width, height = rect
        x2, y2 = x1 + width, y1 + height

        # Calculate the new x', y' in warping coordinate system
        n_x1, n_y1 = cls.map_point([x1, y1], meta)
        n_x2, n_y2 = cls.map_point([x2, y2], meta)
        new_rect = [n_x1, n_y1, n_x2 - n_x1, n_y2 - n_y1]
        return new_rect

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
                dewarped_img, meta = self.predict(img)
                single_result["output"] = dewarped_img
                single_result["confidence"] = 0.99
                single_result["metadata"] = meta
                results.append(single_result)
            except Exception as e:
                logger.error("Error warping image: {}".format(str(e)))
                single_result["output"] = img
                single_result["confidence"] = None
                single_result["metadata"] = {}
                results.append(single_result)
        gc.collect()
        return results



if __name__ == "__main__":
    """ Unit test """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', type=str, default='shufllenetv2')
    parser.add_argument('--input_channel', type=int, default=3)
    parser.add_argument('--output_channel', type=int, default=256)
    parser.add_argument('--num_refinement_stages', type=int, default=2)
    parser.add_argument('--num_heatmaps', type=int, default=6)
    parser.add_argument('--num_pafs', type=int, default=18)
    opt = parser.parse_known_args()[0]

    image_folder = "./assets/Invoice_Toyota4_CameraData_20191224/images"
    debug_folder = "./assets/debug"
    # model_path = "./corner_weights/Dewarp/best_loss.pt"
    model_path = "./best_loss.pt"

    if os.path.exists(debug_folder):
        shutil.rmtree(debug_folder)
    os.makedirs(debug_folder)

    list_files = list(map(lambda f: \
        os.path.join(image_folder, f), os.listdir(image_folder)))
    model = DewarpModel(opt, model_path)

    for fp in list_files:
        print(fp)
        output = model.process([fp])
        cv2.imwrite(os.path.join(debug_folder, \
            os.path.basename(fp)), output[0]["output"])