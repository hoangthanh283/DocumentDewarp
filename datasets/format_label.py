import os
import cv2
import json 
import copy
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt

LABEL_POSTFIX = "labels"
IMAGE_POSTFIX = "images"


class FormatLabel(object):
    """ Format QA label to key-point label 
    Args:
        img_size (int): size to resize image 
        num_keypoints (int): number of keypoints
    """
    def __init__(self, img_size=256, num_keypoints=8):
        self.img_size = img_size
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
        ratio = self.img_size / max(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)

        # Resize image & copute padding if needed
        resized_image = cv2.resize(image, (new_w, new_h))
        h_padding = (self.img_size - resized_image.shape[0]) // 2
        w_padding = (self.img_size - resized_image.shape[1]) // 2

        padding_image = cv2.copyMakeBorder(
            resized_image,
            top=h_padding,
            bottom=h_padding,
            left=w_padding,
            right=w_padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0])
        meta = {"ratio": ratio, "padding": [h_padding, w_padding]}
        return (padding_image, meta)

    def get_key_points(self, image, locations):
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
                    point_x = int(point_x * ratio + w_padding)
                    point_y = int(point_y * ratio + h_padding)
                    all_corners.append([point_x, point_y])

        if len(all_corners) == 0:
            return (None, new_img)
        # Get all the extreme corner points 
        all_corners = np.array(all_corners)
        ex_corners = cv2.convexHull(all_corners) # compute the contour
        epsilon = 0.05 * cv2.arcLength(ex_corners, True) # epsilon may need to be tweaked
        approx_hull = cv2.approxPolyDP(ex_corners, epsilon, True)

        ex_corners = ex_corners.squeeze(1)
        approx_hull = approx_hull.squeeze(1)

        # Check that there are only 4 points left in the approximate hull 
        # ,if there are more, then need to increase epsilon
        debug_img = copy.deepcopy(new_img)
        for final_point in approx_hull:
            cv2.circle(debug_img, \
                (final_point[0], final_point[1]),5,(0, 255, 255), 5)
        cv2.drawContours(debug_img, [ex_corners], 0, (0, 255, 0), 5)
        return (ex_corners.tolist(), debug_img)

    def generate_annotations(self, key_points, image_name):
        """ Generate annotations for keypoints 
        Args:
            key_points (lst): list of keypoint positions 
        Return:
            annotation (dict): dict of annotation
        """
        assert type(key_points) == list, "Only accept list type!"
        bbox = list(cv2.boundingRect(np.array(key_points)))

        # Add 0: occuled, 1: visible, 2: not labeled (x=y=0)
        key_points = list(map(lambda p: [p[0], p[1], 1], key_points))
        # key_points.extend((NUM_KEYPOINT - len(key_points)*[0, 0, 2]))
        # flatten_key_points = list(itertools.chain.from_iterable(key_points))

        # Center of doc with given bounding box
        doc_center = [
            bbox[0] + bbox[2] / 2, 
            bbox[1] + bbox[3] / 2
        ]

        annotation = {
            "img_paths": image_name,
            "img_width": self.img_size, 
            "img_height": self.img_size,
            "objpos": doc_center,
            "bbox": bbox,
            "keypoints": key_points, 
            "num_keypoints": len(key_points),
            "segmentations": [],
            "scale_provided": bbox[3] / self.img_size,
            "processed_other_annotations": []
        } 
        return annotation

    def process(self, input_folder):
        """ Process label format to keypoint format 
        Args:
            input_folder (str): input directory of label folders, 
            contains images, labels folders
        Return:
            None
        """
        debug_dir = os.path.join(input_folder, "debug")
        annotation_file = os.path.join(input_folder, "annotation.pkl")
        if not os.path.exists(debug_dir):
            os.mkdir(debug_dir)

        prepared_annotations = []
        label_dir = os.path.join(input_folder, LABEL_POSTFIX)
        list_labels = list(map(lambda p: \
            os.path.join(label_dir, p), os.listdir(label_dir)))

        for filename in list_labels:
            print(os.path.basename(filename))
            with open(filename, 'r', encoding='utf-8') as jsonfile:
                data = json.load(jsonfile)
                locations = data["attributes"]["_via_img_metadata"]["regions"]
                image_name = data["file_name"]
                image = cv2.imread(os.path.join(\
                    input_folder, IMAGE_POSTFIX, image_name), cv2.IMREAD_COLOR)

                # Generate key-points
                key_points, debug_img = self.get_key_points(image, locations)
                cv2.imwrite(os.path.join(debug_dir, image_name), debug_img)
                
                # Get annotations from keypoints
                if key_points != None:
                    anns = self.generate_annotations(key_points, image_name)
                    prepared_annotations.append(anns)
        
        with open(annotation_file, 'wb') as f:
            pickle.dump(prepared_annotations, f)
        return annotation_file



if __name__ == "__main__":
    """ Unit test """
    root_dir = "./assets/dewarp_labeled_newcameradata"
    list_folders = list(map(lambda p: \
        os.path.join(root_dir, p), os.listdir(root_dir)))
    
    # Define format label instance
    format_labeler = FormatLabel(img_size=512, num_keypoints=4)

    for folder_path in list_folders:
        out = format_labeler.process(folder_path)
    

            
