import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from forpt.utils.logger import get_root_logger
from forpt.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class ForInstanceV2Dataset(DefaultDataset):
    
    def __init__(
        self,
        lr_file=None,
        **kwargs,
    ):
        self.lr = np.atleast_1d(np.loadtxt(lr_file, dtype=str)) if lr_file is not None else None
        super().__init__(**kwargs)

    def get_data_list(self):
        if self.lr is None:
            data_list = super().get_data_list()
        else:
            print('lr: ', self.lr)
            data_list = [
                os.path.join(self.data_root, "train", name) for name in self.lr
            ]
        return data_list