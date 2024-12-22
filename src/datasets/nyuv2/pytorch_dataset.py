# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

import cv2
import numpy as np

from ..dataset_base import DatasetBase
from .nyuv2 import NYUv2Base


class NYUv2(NYUv2Base, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 n_classes=40,
                 split='train',
                 depth_mode='refined',
                 with_input_orig=False):
        super(NYUv2, self).__init__()
        assert split in self.SPLITS
        assert n_classes in self.N_CLASSES
        assert depth_mode in ['refined', 'raw']

        self._n_classes = n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self._cameras = ['kv1']

        self.path = '/kaggle/input/'
        self._filenames = self.get_files_by_extension(self.path)

        # load class names
        self._class_names = getattr(self, f'CLASS_NAMES_{self._n_classes}')

        # load class colors
        self._class_colors = np.array(
            getattr(self, f'CLASS_COLORS_{self._n_classes}'),
            dtype='uint8'
        )

        # note that mean and std differ depending on the selected depth_mode
        # however, the impact is marginal, therefore, we decided to use the
        # stats for refined depth for both cases
        # stats for raw: mean: 2769.0187903686697, std: 1350.4174149841133
        self._depth_mean = 2841.94941272766
        self._depth_std = 1417.2594281672277

    def get_files_by_extension(self, path):
        file_list = []
        subpaths = os.listdir(self.path + 'hypersimrgb/downloads')
        subpaths.sort()
        if self._split == 'train':
            subpaths = subpaths[100:350]
        else:
            subpaths = subpaths[400:]
        for folder in subpaths:
            folders = os.listdir(self.path + 'hypersimrgb/downloads/' + folder + '/images/')
            for f in folders:
                files = os.listdir(self.path + 'hypersimrgb/downloads/' + folder + '/images/' + f)
                for fi in files:
                    rgb_path = self.path + 'hypersimrgb/downloads/' + folder + '/images/' + f + '/' + fi
                    depth_path = self.path + 'hyperdepth/downloads/' + folder + '/images/' + f[:len(f)-13] + 'geometry_preview/' + fi[:len(fi)-9]+'depth_meters.png'
                    labels_path = self.path + 'hyperlabels/downloads/' + folder + '/images/' + f[:len(f)-13] + 'geometry_hdf5/' + fi[:len(fi)-9]+'semantic.hdf5'
                    file_list.append([rgb_path, depth_path, labels_path])
        return file_list

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes + 1

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def _load(self, directory, filename):
        im = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if im.ndim == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return im

    def load_image(self, idx):
        return self._load(self._filenames[idx][0])

    def load_depth(self, idx):
        if self._depth_mode == 'raw':
            return self._load(self._filenames[idx][1])
        else:
            return self._load(self._filenames[idx][1])

    def load_label(self, idx):
        x = h5py.File(self._filenames[idx][2])
        return x['dataset'][:]

    def __len__(self):
        return len(self._filenames)
