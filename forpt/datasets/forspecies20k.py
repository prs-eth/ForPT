import os
import numpy as np
import pointops
import torch
from torch.utils.data import Dataset
from copy import deepcopy


from forpt.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


@DATASETS.register_module()
class ForSpecies20KDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/forspecies20k/processed",
        class_names=None,
        transform=None,
        num_points=8192,
        uniform_sampling=True,
        save_record=True,
        test_mode=False,
        test_cfg=None,
        loop=1,
    ):
        super().__init__()
        self.data_root = data_root
        self.class_names = dict(zip(class_names, range(len(class_names))))
        self.split = split
        self.num_point = num_points
        self.uniform_sampling = uniform_sampling
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        # if test_mode:
        #     self.post_transform = Compose(self.test_cfg.post_transform)
        #     self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

        self.data = {}

    def get_data(self, idx):
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx]
        if data_name in self.data.keys():
            return self.data[data_name]
        else:
            data_shape = "_".join(data_name.split("_")[0:-1])
            data_path = os.path.join(
                self.data_root, data_shape, self.data_list[data_idx] + ".txt"
            )
            data = np.loadtxt(data_path, delimiter=",").astype(np.float32)
            if self.num_point is not None:
                if self.uniform_sampling:
                    with torch.no_grad():
                        mask = pointops.farthest_point_sampling(
                            torch.tensor(data).float().cuda(),
                            torch.tensor([len(data)]).long().cuda(),
                            torch.tensor([self.num_point]).long().cuda(),
                        )
                    data = data[mask.cpu()]
                else:
                    data = data[: self.num_point]
            coord = data[:, 0:3]
            category = np.array([self.class_names.get(data_shape, -1)])
            return dict(coord=coord, category=category)

    def get_data_list(self):
        assert isinstance(self.split, str)
        split_path = os.path.join(
            self.data_root, "{}.txt".format(self.split)
        )
        data_list = np.loadtxt(split_path, dtype="str")
        return data_list

    def get_data_name(self, idx):
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx]

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop

    def prepare_train_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        data_dict['tree_id'] = int(self.get_data_name(idx).split("_")[-1])
        return data_dict
        
