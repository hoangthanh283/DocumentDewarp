import cv2
import json
import math
import argparse
import numpy as np



class Saver(object):
    """ Checkpoint class """
    def save_checkpoint(self, state, filepath='model.pt'):
        """ Save model checkpoints
        Args:
            state (model dict): model's state for saving
            filename (str): Checkpoint file path 
        Returns:
            filepath (str): model path 
        """
        torch.save(state, filepath)
        return filepath

    def load_checkpoint(self, filepath):
        """ Load model checkpoints
        Args:
            filepath (str): checkpoint file path
        Returns:

        """
        if DEVICE == torch.device('cpu'):
            checkpoint = torch.load(filepath, \
                map_location=lambda storage, loc: storage)
        else:
            checkpoint = torch.load(filepath)
        return checkpoint


class Averager(object):
    """Compute average for torch.Tensor, used for loss average."""
    def __init__(self):
        self.reset()

    def add(self, v):
        self.n_count += 1
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def average(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img


def pad_width(img, stride, pad_value, min_dims):
    h, w = img.shape[:2]
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(img, pad[0], pad[2], pad[1], pad[3], \
        cv2.BORDER_CONSTANT, value=pad_value)
    return padded_img, pad

def four_point_transform(image, pts):
    if pts == None:
        return image
    pts = np.asarray(pts ,dtype=np.float32)
    (tl, tr, br, bl) = pts

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR)
    return (warped, M)
