from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms


class KITTIDataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(KITTIDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, '')
        self.image_dir = os.path.join(self.root, 'images')
        self.label_dir = os.path.join(self.root, 'gt')
        self.edge_dir = os.path.join(self.root, 'edge_binary')

        file_list = os.path.join(self.root, "", self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        edge_path = os.path.join(self.edge_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        edge = np.asarray(Image.open(edge_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id, edge


class BDDAugDataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 2
        self.palette = palette.get_voc_palette(self.num_classes)
        super(BDDAugDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'VOCdevkit/VOC2012')

        file_list = os.path.join(self.root, "ImageSets/Segmentation", self.split + ".txt")
        file_list = [line.rstrip().split(' ') for line in tuple(open(file_list, "r"))]
        self.files, self.labels = list(zip(*file_list))

    def _load_data(self, index):
        image_path = os.path.join(self.root, self.files[index][1:])
        label_path = os.path.join(self.root, self.labels[index][1:])
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id


class KITTI(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=0,
                 val=False,
                 shuffle=False, flip=False, rotate=False, blur=False, augment=False, val_split=None, return_id=False):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        if split in ["train_aug", "trainval_aug", "val_aug", "test_aug"]:
            self.dataset = BDDAugDataset(**kwargs)
        elif split in ["train", "trainval", "val", "test"]:
            self.dataset = KITTIDataset(**kwargs)
        else:
            raise ValueError(f"Invalid split name {split}")
        super(KITTI, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
