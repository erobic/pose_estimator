import numpy as np
from config import config
import time
from PIL import Image

eps = 1e-5  # avoid divide by zero error
conf = config()


def generate_deltas(num_features=conf['num_features']):
    """
    Deltas define the offsets for depth comparison
    :param num_features:
    :return:
    """
    # TODO: experiment with the limiting values for delta
    delta1 = {'x': np.random.uniform(0, conf['width'] / 5, size=num_features),
              'y': np.random.uniform(0, conf['height'] / 5, size=num_features)}
    delta2 = {'x': np.random.uniform(0, conf['width'] / 5, size=num_features),
              'y': np.random.uniform(0, conf['height'] / 5, size=num_features)}
    # __show_deltas(delta1)
    # __show_deltas(delta2)
    return delta1, delta2


def generate_thresholds(num_thresholds=conf['num_thresholds'], num_features=conf['num_features']):
    """
    (Current unused). Generates thresholds to compare depth features
    :param num_thresholds:
    :param num_features:
    :return:
    """
    thresholds = []
    for i in range(0, num_features):
        t = np.random.uniform(0, 1, size=num_thresholds)
        thresholds.append(t)
    return thresholds


def __show_deltas(deltas):
    img = Image.new('RGB', (conf['width'], conf['height']))
    for x, y in zip(deltas['x'], deltas['y']):
        img.putpixel((int(x), int(y)), (255, 255, 255))
    print("Showing deltas")
    img.show()


class DepthFeatureExtractor:
    def __init__(self, conf=config()):
        self.conf = conf
        self.x_ind = self.x_indices(conf['width'], conf['height']).T
        self.y_ind = self.y_indices(conf['width'], conf['height']).T

    @staticmethod
    def x_indices(width, height):
        r = np.array([np.arange(width)])
        indices = np.repeat(r, height, axis=0)
        return indices

    @staticmethod
    def y_indices(width, height):
        c = np.array([np.arange(height)]).T
        indices = np.repeat(c, width, axis=1)
        return indices

    def __calc_offset(self, depth_inv, delta, batch_size):
        offset = np.outer(delta, depth_inv)
        offset = offset.reshape(self.conf['num_features'], batch_size, self.conf['width'], self.conf['height'])
        offset = np.swapaxes(offset, 0, 1)
        return offset.astype(int)

    def __calc_x_offset(self, depth_inv, delta_x, batch_size):
        x_offset = self.__calc_offset(depth_inv, delta_x, batch_size)
        x_offset += self.x_ind
        return x_offset

    def __calc_y_offset(self, depth_inv, delta_y, batch_size):
        y_offset = self.__calc_offset(depth_inv, delta_y, batch_size)
        y_offset += self.y_ind
        return y_offset

    def __calc_offset_for_selected(self, initial_offset, depth_inv, delta, batch_size, num_selections):
        """

        :return: B X Num features X Num selections
        """
        offset = np.outer(delta, depth_inv)
        offset = offset.reshape(self.conf['num_features'], batch_size, num_selections)
        offset += initial_offset
        offset = np.swapaxes(offset, 0, 1)
        # print("offset.shape = ", offset.shape)
        return offset.astype(int)

    def extract_depth_features_for_selected(self, img_batch, delta1, delta2, selected_coords_batch=None):
        """
        Extracts depth features for selected pixels.
        (B = Batch size, F = No. of features, W = Width, H = Height)
        :param: selected_px_coords: Coordinates for selected pixels. Format: [[row_numbers][column numbers]] i.e. the output from np.nonzero() or np.where()
        :param img_batch: Batch of images with depth values. Shape: B X W X H
        :param delta1
        :param delta2
        :return: features: B X F X Num selections
        """
        batch_size = img_batch.shape[0]

        if selected_coords_batch is not None:

            num_selections = selected_coords_batch.shape[2]
            # Get selected depths
            selected_batch = []
            for i in range(0, batch_size):
                img = img_batch[i]
                selected_coord = selected_coords_batch[i]
                selected_depths = img[selected_coord[0], selected_coord[1]]
                selected_batch.append(selected_depths)
            selected_batch = np.array(selected_batch)

            # selected_batch[selected_batch < 1e-6] = 1e-2 # TODO: Check range of depth values

            depth_inv = 1. / (selected_batch + 1e-4)

            selected_coords_batch = np.swapaxes(selected_coords_batch, 0, 1)  # 2 X B X Num selections
            x1 = self.__calc_offset_for_selected(selected_coords_batch[0], depth_inv, delta1['x'], batch_size,
                                                 num_selections)
            y1 = self.__calc_offset_for_selected(selected_coords_batch[1], depth_inv, delta1['y'], batch_size,
                                                 num_selections)
            x2 = self.__calc_offset_for_selected(selected_coords_batch[0], depth_inv, delta2['x'], batch_size,
                                                 num_selections)
            y2 = self.__calc_offset_for_selected(selected_coords_batch[1], depth_inv, delta2['y'], batch_size,
                                                 num_selections)

        else:
            depth_inv = 1 / (img_batch + 1e-4)

            # Calculate (x, y) coordinates of target pixels
            x1 = self.__calc_x_offset(depth_inv, delta1['x'], batch_size)
            y1 = self.__calc_y_offset(depth_inv, delta1['y'], batch_size)
            x2 = self.__calc_x_offset(depth_inv, delta2['x'], batch_size)
            y2 = self.__calc_y_offset(depth_inv, delta2['y'], batch_size)

        # Remove elements referring to pixels outside the image
        x1_mask = np.logical_or(x1 < 0, x1 >= self.conf['width'])
        x2_mask = np.logical_or(x2 < 0, x2 >= self.conf['width'])
        y1_mask = np.logical_or(y1 < 0, y1 >= self.conf['height'])
        y2_mask = np.logical_or(y2 < 0, y2 >= self.conf['height'])
        x1[x1_mask] = 0
        x2[x2_mask] = 0
        y1[y1_mask] = 0
        y2[y2_mask] = 0

        # Now compute the depth features
        batch_output = []

        for img_no in range(0, x1.shape[0]):
            curr_img_feats = []
            for feat_no in range(0, x1.shape[1]):
                curr_img = img_batch[img_no]
                curr_x1 = x1[img_no][feat_no]
                curr_y1 = y1[img_no][feat_no]
                curr_x2 = x2[img_no][feat_no]
                curr_y2 = y2[img_no][feat_no]
                target_img1 = curr_img[curr_x1, curr_y1]
                target_img2 = curr_img[curr_x2, curr_y2]
                feat = target_img1 - target_img2
                curr_img_feats.append(feat)
            batch_output.append(curr_img_feats)

        batch_output = np.array(batch_output)

        return batch_output
