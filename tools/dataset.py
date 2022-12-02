import os
import pandas as pd
import torch
import numpy as np
import json
from torch.utils.data import Dataset
import utils.python.image3d_io as cio
import utils.python.image3d_tools as ctools
from .crop_thick import *
import pandas as pd
from .normalizers import AdaptiveNormalizer
from torch.utils.data.sampler import Sampler
from itertools import chain
import random


def read_train_csv(csv_file, with_label=True):
    t1_list, t2_list, mask_list, label_list = [], [], [], []
    df = pd.read_csv(csv_file, header=None)
    if sum(df[1].isnull()) > 0:
        flag = 0
    else:
        flag = 1
    for i in range(len(df)):
        row = df.iloc[i]
        t1_path = row[0]
        assert (os.path.isfile(t1_path))
        if len(row) <= 2:
            mask_path = None
        else:
            mask_path = row[2]
            if isinstance(mask_path, str):
                assert (os.path.isfile(mask_path))
        if flag:
            t2_path = row[1]
            assert (os.path.isfile(t2_path))
            t2_list.append(t2_path)
        t1_list.append(t1_path)
        mask_list.append(mask_path)
        if with_label is True:
            label = int(row[3])
            label_list.append(label)
    return t1_list, t2_list, mask_list, label_list


axes = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]], dtype=np.float32)


def resample_thick_image(images, spacing, crop_size, default_values, interpolation='NN'):
    new_spacing = [spacing[0], spacing[1], images[0].spacing()[2]]
    frame = images[0].frame()
    target_size = np.ceil(images[0].size() / np.array(new_spacing) * images[0].spacing())
    if crop_size is not None:
        target_size[:2] = np.array((target_size[:2], np.array(crop_size))).max(0)
    frame.set_spacing(new_spacing)
    frame.set_axes(axes)
    for i in range(len(images)):
        images[i].set_axes(axes)
        if interpolation == 'NN':
            images[i] = ctools.resample_nn(images[i], frame, target_size, default_value=default_values[i])
        else:
            images[i] = ctools.resample_trilinear(images[i], frame, target_size, default_value=default_values[i])
    return images


def crop_mask_norm(img, mask, crop=False):
    bound = ctools.bounding_box_voxel(mask, 1, 4)
    if crop is True:
        gap0 = max(192 - bound[1][0] + bound[0][0], 10)
        gap1 = max(192 - bound[1][1] + bound[0][1], 10)
        lower = bound[0] - np.array([gap0, gap1, 0])
        upper = bound[1] + np.array([gap0, gap1, 0])
        img_crop = ctools.crop(img, lower, upper)
        mask_crop = ctools.crop(mask, lower, upper)
        i = img_crop.to_numpy()
        l = mask_crop.to_numpy()
        i[l == 0] = -1
    else:
        i = img.to_numpy()
        l = mask.to_numpy()
        #print('before: ', l.max())
        i[l == 0] = -1
        l[l == 1] = 0
        l[l > 0] = 1
        #print('after: ', l.max())
    return i, l


def bound_2d(t):
    ind = np.where(t > 0)
    if np.sum(t > 0) == 0:
        return 0, t.shape[0], 0, t.shape[1]
    xmin, xmax, ymin, ymax = min(ind[0]), max(ind[0]), min(ind[1]), max(ind[1])
    return xmin, xmax, ymin, ymax


def random_crop_2d(t, target_size, rand=[5, 5]):
    xmin, xmax, ymin, ymax = bound_2d(t)
    center_x = int(xmin / 2 + xmax / 2)
    center_y = int(ymin / 2 + ymax / 2)
    trans = np.random.uniform(-1 * rand, rand, size=[2]).astype(np.int16)
    center_x += trans[0]
    center_y += trans[1]
    x_lower = max(0, center_x - target_size[0] // 2)
    x_upper = min(t.shape[0], center_x + target_size[0] // 2)
    y_lower = max(0, center_y - target_size[1] // 2)
    y_upper = min(t.shape[1], center_y + target_size[1] // 2)
    return x_lower, x_upper, y_lower, y_upper


class ClassificationDataset(Dataset):
    """ training data set for volumetric segmentation """

    def __init__(self, imlist_file, num_classes, spacing, crop_size, bag_size, default_values,
                 random_translation,
                 interpolation, crop_normalizers,
                 with_label=True
                ):
        self.t1_list, self.t2_list, self.mask_list, self.label_list = read_train_csv(imlist_file, with_label)
        assert len(self.t1_list) == len(self.mask_list)
        if with_label is True:
            assert len(self.t1_list) == len(self.label_list)
        assert len(self.t2_list) == 0 or len(self.t2_list) == len(self.t1_list)
        self.num_classes = num_classes
        self.default_values = default_values
        self.bag_size = bag_size
        assert self.bag_size > 0
        self.spacing = np.array(spacing, dtype=np.double)
        assert self.spacing.size == 2, 'only 2-element of spacing is supported'
        if crop_size is not None:
            self.crop_size = np.array(crop_size, dtype=np.int32)
            assert self.crop_size.size == 2, 'only 2-element of crop size is supported'
        else:
            self.crop_size = None
        self.random_translation = np.array(random_translation, dtype=np.double)
        assert self.random_translation.size == 2, 'Only 2-element of random translation is supported'
        assert interpolation in ('LINEAR', 'NN'), 'interpolation must either be a LINEAR or NN'
        self.interpolation = interpolation
        assert isinstance(crop_normalizers, list), 'crop normalizers must be a list'
        self.crop_normalizers = crop_normalizers
        self.with_label = with_label

    def num_modality(self):
        """ get the number of input image modalities """
        if len(self.t2_list) == len(self.t1_list):
            return 2
        return 1

    def __len__(self):
        """ get the number of images in this data set """
        return len(self.t1_list)

    def _random_crop_center2d(self, images, mask):
        if self.crop_size is not None:
            crop_size = self.crop_size
        else:
            crop_size = mask.size()[:2]
        mask = mask.to_numpy()
        mask[mask > 0] = 1
        if len(images) == 2:
            t1 = images[0].to_numpy()
            t2 = images[1].to_numpy()
        else:
            t1 = images[0].to_numpy()
            t2 = None
        newimg1 = np.zeros([self.bag_size, 1, crop_size[1], crop_size[0]]) - 1
        newtissue = np.zeros([self.bag_size, 1, crop_size[1], crop_size[0]])
        if t2 is not None:
            newimg2 = np.zeros([self.bag_size, 1, crop_size[1], crop_size[0]]) - 1

        for i in range(mask.shape[0]):
            t = mask[i]
            i1 = t1[i]
            if t2 is not None:
                i2 = t2[i]
            x_lower, x_upper, y_lower, y_upper = random_crop_2d(t, crop_size, rand=self.random_translation)
            newimg1[i, 0, :(x_upper - x_lower), :(y_upper - y_lower)] = i1[x_lower:x_upper, y_lower:y_upper]
            newtissue[i, 0, :(x_upper - x_lower), :(y_upper - y_lower)] = t[x_lower:x_upper, y_lower:y_upper]
            if t2 is not None:
                newimg2[i, 0, :(x_upper - x_lower), :(y_upper - y_lower)] = i2[x_lower:x_upper, y_lower:y_upper]

        if t2 is not None:
            return newimg1, newimg2, newtissue
        return newimg1, None, newtissue

    def __getitem__(self, index):
        t1_path, mask_path = self.t1_list[index], self.mask_list[index]
        if self.with_label:
            label = self.label_list[index]

        images = [cio.read_image(t1_path,  dtype=np.float32)]
        if len(self.t2_list) == len(self.t1_list):
            t2_path = self.t2_list[index]
            images.append(cio.read_image(t2_path,  dtype=np.float32))

        if isinstance(mask_path, str):
            mask = cio.read_image(mask_path, dtype=np.int16)
        else:
            mask = images[0]
            m = mask.to_numpy()
            m[m != 0] = 1
            mask.from_numpy(m)

        for idx in range(len(images)):
            if self.crop_normalizers[idx] is not None:
                self.crop_normalizers[idx](images[idx])

        images = resample_thick_image(images,
                                      spacing=self.spacing,
                                      crop_size=self.crop_size,
                                      default_values=self.default_values,
                                      interpolation=self.interpolation)
        mask = resample_thick_image([mask],
                                    spacing=self.spacing,
                                    crop_size=self.crop_size,
                                    default_values=self.default_values)[0]

        t1, t2, mask = self._random_crop_center2d(images, mask)
        if self.with_label and t2 is not None:
            samples = {'t1': t1,
                       't2': t2,
                       'label': label,
                       'mask': mask}
        elif t2 is not None:
            samples = {'t1': t1,
                       't2': t2,
                       'mask': mask}
        elif self.with_label:
            samples = {'t1': t1,
                       'mask': mask,
                       'label': label}
        else:
            samples = {'t1': t1,
                       'mask': mask}

        return samples


def class_index(dataset):
    d = {}
    for i, label in enumerate(list(dataset.label_list)):
        # tag = dataset[i][1]
        if not d.get(label):
            d[label] = [i]
        else:
            d[label].append(i)

    return d


class EpochConcateSampler_balance(Sampler):
    """Concatenate  all epoch index arrays into one index array.

    Arguments:
        data_source (Dataset): dataset to sample from
        epoch(int): epoch num
    """

    def __init__(self, epoch, index_dict):
        self.epoch = epoch
        self.index_dict = index_dict
        self.class_length = [len(index_dict[label]) for label in index_dict]
        self.data_length = min(self.class_length) * len(index_dict)

    def cross_list(self, index_dict):
        class_list = [index_dict[label] for label in index_dict]
        tmp = list(chain.from_iterable(zip(*class_list)))
        return tmp

    def __iter__(self):
        index_all = []
        for i in range(self.epoch):
            index_dict = self.index_dict
            for k in index_dict.keys():
                random.shuffle(index_dict[k])
            #print('len:', len(index_pos2))
            tmp = self.cross_list(index_dict)

            random.shuffle(tmp)
            index_all += tmp
        return iter(index_all)

    def __len__(self):
        return self.data_length * self.epoch







