import os
import json 
import copy
import math
import pickle
import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from utils.tools import normalize
from datasets.transformations import (
    ConvertKeypoints, Scale, Rotate, CropPad, Flip
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)  # To prevent freeze of DataLoader
LABEL_POSTFIX = "labels"
IMAGE_POSTFIX = "images"
CORNERS_KPT_IDS = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]



class GetDataLoader(object):
    """ Data loader for training & validation """
    def __init__(self, opt):        
        # Define train data loader
        self.train_dataset = DocumentDataSet(opt, \
            is_train=True, transform=None)
        self.train_loader = DataLoader(self.train_dataset, \
            batch_size=opt.batch_size, \
            shuffle=True, num_workers=opt.num_workers)

        # Define validation data loader
        self.val_dataset = DocumentDataSet(opt, is_train=False)
        self.val_loader = DataLoader(self.val_dataset, \
            batch_size=opt.batch_size, \
            shuffle=False, num_workers=opt.num_workers)


class FormatLabel(object):
    """ Format QA label to key-point label 
    Args:
        img_size (int): size to resize image 
        num_keypoints (int): number of keypoints
    """
    def __init__(self, new_size=(256, 256), num_keypoints=4):
        self.new_size = new_size
        self.num_keypoints = num_keypoints

    def resize_image(self, image):
        """ Resize and padding image with given size 
        Args:
            image (np.array): input numpy array image 
        Return:
            padding_image (np.array): output resized image
            meta (dict): information about ratio resize and padding
        """
        h, w = image.shape[:2]
        new_h, new_w = self.new_size
        
        ratio = max(new_h, new_w) / max(h, w)
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        # Resize image & copute padding if needed
        resized_image = cv2.resize(image, (resize_w, resize_h), cv2.INTER_AREA)
        h_padding = (new_h - resize_h)
        w_padding = (new_w - resize_w)

        padding_image = cv2.copyMakeBorder(
            resized_image,
            top=h_padding//2,
            bottom=h_padding - h_padding//2,
            left=w_padding//2,
            right=w_padding - w_padding//2,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        meta = {"ratio": ratio, "padding": [h_padding, w_padding]}
        return (padding_image, meta)

    def get_key_points(self, image, locations, is_visualize=False):
        """ Define keypoints based document corners 
        Args:
            image (np.array): input image
            locations (list): list of field dicts
        Return:
            ex_corners (lst): list of corner points 
        """
        # Resize image
        new_img, meta = self.resize_image(image)
        ratio = meta["ratio"]
        h_padding, w_padding = meta["padding"]
        all_corners = []
        
        for loc in locations:
            if (loc["shape_attributes"]["name"] == "polygon" \
                and loc["region_attributes"].get("formal_key", None) == "dewarp"):
                all_points_x = loc["shape_attributes"]["all_points_x"]
                all_points_y = loc["shape_attributes"]["all_points_y"]

                min_x = int(min(all_points_x))
                max_x = int(max(all_points_x))
                min_y = int(min(all_points_y))
                max_y = int(max(all_points_y))

                h, w = image.shape[:2]
                assert min_x < w, max_x <= w
                assert min_y < h, max_y <= h

                for (point_x, point_y) in zip(all_points_x, all_points_y):
                    point_x = int(point_x * ratio + w_padding//2)
                    point_y = int(point_y * ratio + h_padding//2)
                    all_corners.append([point_x, point_y])

                # Find the center point
                all_point_positions = np.array([[x, y] \
                    for (x, y) in zip(all_points_x, all_points_y)])
                momentum = cv2.moments(all_point_positions)
                c_x = int(momentum['m10'] / momentum['m00'])
                c_y = int(momentum['m01'] / momentum['m00'])
                
                # scale down the point
                center_point = [\
                    int(c_x * ratio + w_padding//2), \
                    int(c_y * ratio + h_padding//2)]

        if len(all_corners) == 0:
            return (None, new_img)

        # Get all the extreme corner points 
        all_corners = np.array(all_corners)
        ex_corners = cv2.convexHull(all_corners) # compute the contour
        epsilon = 0.05 * cv2.arcLength(ex_corners, True) # epsilon may need to be tweaked
        approx_hull = cv2.approxPolyDP(ex_corners, epsilon, True)

        ex_corners = ex_corners.squeeze(1)
        approx_hull = approx_hull.squeeze(1)
        ex_corners = ex_corners.tolist() + [center_point]

        # Check that there are only 4 points left in the approximate hull 
        # ,if there are more, then need to increase epsilon
        if is_visualize == True:
            debug_img = copy.deepcopy(new_img)
            for final_point in approx_hull:
                cv2.circle(debug_img, \
                    (final_point[0], final_point[1]),5,(0, 255, 255), 5)
            cv2.drawContours(debug_img, [np.array(ex_corners)], 0, (0, 255, 0), 5)
            plt.imshow(debug_img); plt.show()
        return (ex_corners, new_img)

    def generate_annotations(self, key_points, image_name):
        """ Generate annotations for keypoints 
        Args:
            key_points (lst): list of keypoint positions 
        Return:
            annotation (dict): dict of annotation
        """

        if key_points != None:
            assert type(key_points) == list, "Only accept list type!"
            bbox = list(cv2.boundingRect(np.array(key_points)))
            scale_provided = bbox[3] / self.new_size[0]

            # Add 0: occuled, 1: visible, 2: not labeled (x=y=0)
            key_points = list(map(lambda p: [p[0], p[1], 1], key_points))
            # key_points.extend((NUM_KEYPOINT - len(key_points)*[0, 0, 2]))
            # flatten_key_points = list(itertools.chain.from_iterable(key_points))

            # Center of doc with given bounding box
            doc_center = [
                bbox[0] + bbox[2] / 2, 
                bbox[1] + bbox[3] / 2
            ]
        else:
            print("Number of keypoints are 0 with {0}".format(image_name))
            key_points = [[0, 0, 0] for _ in range(self.num_keypoints)]
            bbox = [0, 0, 0, 0]
            doc_center = [0, 0]
            scale_provided = 1

        annotation = {
            "img_paths": image_name,
            "img_width": self.new_size[1], 
            "img_height": self.new_size[0],
            "objpos": doc_center,
            "bbox": bbox,
            "keypoints": key_points, 
            "num_keypoints": len(key_points),
            "segmentations": [],
            "scale_provided": scale_provided,
            "processed_other_annotations": []
        } 
        return annotation

    def process(self, image_folder, label_path):
        """ Process label format to keypoint format 
        Args:
            label_file (str): label files
        Return:
            anns (dict): annotation of key-points
        """
        with open(label_path, 'r', encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
            locations = data["attributes"]["_via_img_metadata"]["regions"]
            image_name = data["file_name"]
            image = cv2.imread(\
                os.path.join(image_folder, image_name), cv2.IMREAD_COLOR)

            # Generate key-points
            key_points, new_img = self.get_key_points(image, locations)
            
            # Get annotations from keypoints
            anns = self.generate_annotations(key_points, image_name)
            return (new_img, anns)

           
class DocumentDataSet(Dataset):
    """ Document data loader for corner points estimation """
    def __init__(self, opt, is_train=False, transform=None):
        super(DocumentDataSet, self).__init__()
        self.opt = opt
        self._transform = transform
        self._sigma = self.opt.sigma
        self._stride = self.opt.stride
        self._paf_thickness = self.opt.paf_thickness
        self.format_labeler = FormatLabel(\
            new_size=[self.opt.img_height, self.opt.img_width], \
            num_keypoints=self.opt.num_heatmaps - 1)

        if is_train == True:
            self.sample_path = self.opt.train_path
        else:
            self.sample_path = self.opt.val_path
        self.label_paths = self._load_label(self.sample_path)

    def _load_label(self, sample_path):
        label_path = os.path.join(sample_path, LABEL_POSTFIX)
        labels = list(map(lambda f: \
            os.path.join(label_path, f), os.listdir(label_path)))
        return labels

    def _add_gaussian(self, keypoint_map, x, y, stride, sigma):
        n_sigma = 4
        tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
        tl[0] = max(tl[0], 0)
        tl[1] = max(tl[1], 0)

        br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
        map_h, map_w = keypoint_map.shape
        br[0] = min(br[0], map_w * stride)
        br[1] = min(br[1], map_h * stride)

        shift = stride / 2 - 0.5
        for map_y in range(tl[1] // stride, br[1] // stride):
            for map_x in range(tl[0] // stride, br[0] // stride):
                d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                     (map_y * stride + shift - y) * (map_y * stride + shift - y)
                exponent = d2 / 2 / sigma / sigma
                if exponent > 4.6052:  # threshold, ln(100), ~0.01
                    continue
                keypoint_map[map_y, map_x] += math.exp(-exponent)
                if keypoint_map[map_y, map_x] > 1:
                    keypoint_map[map_y, map_x] = 1

    def _generate_keypoint_maps(self, sample):
        n_rows, n_cols = sample['image'].shape[:2] # height, width 
        keypoint_maps = np.zeros(shape=(\
            self.opt.num_heatmaps, \
            math.ceil(n_rows / self._stride), \
            math.ceil(n_cols / self._stride)), \
            dtype=np.float32)  # +1 for bg

        label = sample['label']
        for keypoint_idx in range(self.opt.num_heatmaps - 1):
            keypoint = label['keypoints'][keypoint_idx]
            if keypoint[2] <= 1:
                self._add_gaussian(keypoint_maps[keypoint_idx], \
                    keypoint[0], keypoint[1], self._stride, self._sigma)
            for another_annotation in label['processed_other_annotations']:
                keypoint = another_annotation['keypoints'][keypoint_idx]
                if keypoint[2] <= 1:
                    self._add_gaussian(keypoint_maps[keypoint_idx], \
                        keypoint[0], keypoint[1], self._stride, self._sigma)
        keypoint_maps[-1] = 1 - keypoint_maps.max(axis=0)
        return keypoint_maps

    def _set_paf(self, paf_map, x_a, y_a, x_b, y_b, stride, thickness):
        x_a /= stride
        y_a /= stride
        x_b /= stride
        y_b /= stride
        x_ba = x_b - x_a
        y_ba = y_b - y_a
        _, h_map, w_map = paf_map.shape
        x_min = int(max(min(x_a, x_b) - thickness, 0))
        x_max = int(min(max(x_a, x_b) + thickness, w_map))
        y_min = int(max(min(y_a, y_b) - thickness, 0))
        y_max = int(min(max(y_a, y_b) + thickness, h_map))
        norm_ba = (x_ba * x_ba + y_ba * y_ba) ** 0.5
        if norm_ba < 1e-7:  # Same points, no paf
            return
        x_ba /= norm_ba
        y_ba /= norm_ba

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                x_ca = x - x_a
                y_ca = y - y_a
                d = math.fabs(x_ca * y_ba - y_ca * x_ba)
                if d <= thickness:
                    paf_map[0, y, x] = x_ba
                    paf_map[1, y, x] = y_ba

    def _generate_paf_maps(self, sample):
        n_pafs = len(CORNERS_KPT_IDS)
        n_rows, n_cols = sample['image'].shape[:2] # height, width
        paf_maps = np.zeros(shape=(n_pafs * 2, \
            math.ceil(n_rows / self._stride), \
            math.ceil(n_cols / self._stride)), \
            dtype=np.float32)

        label = sample['label']
        for paf_idx in range(n_pafs):
            keypoint_a = label['keypoints'][CORNERS_KPT_IDS[paf_idx][0]]
            keypoint_b = label['keypoints'][CORNERS_KPT_IDS[paf_idx][1]]
            if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                              keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                              self._stride, self._paf_thickness)
            for another_annotation in label['processed_other_annotations']:
                keypoint_a = another_annotation['keypoints'][CORNERS_KPT_IDS[paf_idx][0]]
                keypoint_b = another_annotation['keypoints'][CORNERS_KPT_IDS[paf_idx][1]]
                if keypoint_a[2] <= 1 and keypoint_b[2] <= 1:
                    self._set_paf(paf_maps[paf_idx * 2:paf_idx * 2 + 2],
                                  keypoint_a[0], keypoint_a[1], keypoint_b[0], keypoint_b[1],
                                  self._stride, self._paf_thickness)
        return paf_maps

    def _transform_sample(self, sample):
        if self._transform:
            sample = self._transform(sample)
        
        # Generate key-point masks
        keypoint_maps = self._generate_keypoint_maps(sample)
        sample['keypoint_maps'] = keypoint_maps
        keypoint_mask = np.zeros(shape=keypoint_maps.shape, dtype=np.float32)

        mask = cv2.resize(sample['mask'], \
            dsize=keypoint_mask.shape[1:], interpolation=cv2.INTER_AREA)

        for idx in range(keypoint_mask.shape[0]):
            keypoint_mask[idx] = mask
        sample['keypoint_mask'] = keypoint_mask

        # Generate paf masks
        paf_maps = self._generate_paf_maps(sample)
        sample['paf_maps'] = paf_maps
        paf_mask = np.zeros(shape=paf_maps.shape, dtype=np.float32)
        
        for idx in range(paf_mask.shape[0]):
            paf_mask[idx] = mask
        sample['paf_mask'] = paf_mask

        # Normalize input image to (-0.5, 0.5)
        image = sample['image'].astype(np.float32)
        # image = (image - 128) / 256
        image = normalize(image, img_mean=128, img_scale=1/256)
        sample['image'] = image.transpose((2, 0, 1))
        return sample

    def __getitem__(self, idx):
        label_file = self.label_paths[idx]
        image_folder = os.path.join(self.sample_path, IMAGE_POSTFIX)
        image, label = self.format_labeler.process(image_folder, label_file)
        mask = np.ones(shape=(label['img_height'], \
            label['img_width']), dtype=np.float32)
        
        sample = {
            'label': label,
            'image': image,
            'mask': mask
        }
        sample = self._transform_sample(sample)
        return sample

    def __len__(self):
        return len(self.label_paths)
    

            
