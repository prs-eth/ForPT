import os
import numpy as np
import pandas as pd
import pointops
import torch
from torch.utils.data import Dataset
from copy import deepcopy

from forpt.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose

@DATASETS.register_module()
class ForAgeDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_root="data/forage",
        metadata_file="FORage_tree_metadata_train_val.csv",
        transform=None,
        num_points=8192,
        uniform_sampling=True,
        save_record=True,
        test_mode=False,
        inference_mode=False,
        test_cfg=None,
        loop=1,
        pc_folder="pc",  # folder containing point cloud files
        target_col="age",   # column in CSV for the regression target
        sample_list_file=None,  # txt list restricting the split to a subset of samples
    ):
        """
        Args:
            split: "train", "val", or "test" (or however you want to separate data).
            data_root: Base directory containing CSV and point cloud folder.
            metadata_file: Name of the metadata CSV with file paths & regression targets.
                The hidden test set has no target column; its targets are then NaN.
            transform: List of data transforms (Compose).
            num_points: If not None, limit or sample the point cloud to this many points.
            uniform_sampling: If True, uses farthest point sampling to pick points.
            save_record: If True, caches the loaded point clouds in a .pth file.
            test_mode: If True, calls `prepare_test_data`.
            inference_mode: If True, calls `prepare_inference_data`.
            test_cfg: Configuration for test-time augmentations or transformations.
            loop: Repeat the dataset N times in training.
            pc_folder: Subfolder inside data_root with one <name>/coord.npy per tree.
            target_col: The name of the column in the CSV for the regression target.
            sample_list_file: Optional .txt list with one sample name per line
                (the `file` value, e.g. `evo_23_ALSHD_1001_119_pine_56.laz`).
                Used for the data-efficiency experiments: the split is restricted
                to the listed samples. The lists are produced offline, nested
                across ratios, by age_gen_subset.py.
        """
        super().__init__()
        self.data_root = data_root
        self.metadata_file = metadata_file
        self.transform = Compose(transform)
        self.num_point = num_points
        self.uniform_sampling = uniform_sampling
        self.loop = loop if not test_mode else 1
        self.test_mode = test_mode
        self.inference_mode = inference_mode
        self.test_cfg = test_cfg if test_mode else None
        self.pc_folder = pc_folder
        self.target_col = target_col
        self.sample_list_file = sample_list_file



        # Build post / aug transforms if test_mode
        if self.test_mode and self.test_cfg is not None:
            print(f"[DEBUG] test_cfg = {test_cfg}")  # For debugging

            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]
        else:
            self.post_transform = None
            self.aug_transform = []

        # Read CSV into a DataFrame
        self.metadata_path = os.path.join(self.data_root, self.metadata_file)
        self.df = pd.read_csv(self.metadata_path)
        # Optional: filter rows based on `split` column if you have it in the CSV
        if "split" in self.df.columns:
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        ### check whether coord file exist or not

        valid_files = os.listdir(os.path.join(self.data_root, self.pc_folder))
        valid_files = [file + '.laz' for file in valid_files]
        self.df = self.df[self.df["file"].isin(valid_files)].reset_index(drop=True)

        ### data-efficiency: keep only the samples listed in sample_list_file
        if self.sample_list_file is not None:
            subset = np.atleast_1d(np.loadtxt(self.sample_list_file, dtype=str))
            num_full = len(self.df)
            self.df = self.df[self.df["file"].isin(set(subset))].reset_index(drop=True)
            assert len(self.df) == len(subset), (
                f"{len(subset)} names listed in {self.sample_list_file} but "
                f"{len(self.df)}/{num_full} matched in the '{split}' split"
            )


        # Create a list of (filename, target_value) to load
        self.data_list = self._get_data_list()

        # Print dataset size
        logger = get_root_logger()
        subset_info = (
            f" (subset {os.path.basename(self.sample_list_file)})"
            if self.sample_list_file is not None
            else ""
        )
        logger.info(
            f"Found {len(self.data_list)} x {self.loop} samples in {split} set{subset_info}."
        )

        # Optionally load / save preprocessed data in a .pth record.
        # The record name is keyed on the full split (not on the subset) so that a
        # data-efficiency run reuses the existing full-split cache -- it is a
        # superset and lookups go by data_name. A subset run must never overwrite
        # that cache with its own partial one, hence the save_record guard below.
        record_name = f"forage_{split}"
        if num_points is not None:
            record_name += f"_{num_points}points"
            if uniform_sampling:
                record_name += "_uniform"
        record_path = os.path.join(self.data_root, f"{record_name}.pth")

        # self.data_cache = {}
        if os.path.isfile(record_path):
            logger.info(f"Loading record: {record_name} ...")
            self.data_cache = torch.load(record_path)
        else:
            logger.info(f"Preparing record: {record_name} ...")
            self.data_cache = {}
            for idx in range(len(self.data_list)):
                data_name = self.data_list[idx][0]  # e.g. filename
                logger.info(f"Parsing data [{idx}/{len(self.data_list)}]: {data_name}")
                self.data_cache[data_name] = self._load_data(idx)
            if save_record and self.sample_list_file is None:
                torch.save(self.data_cache, record_path)

    def _get_data_list(self):
        """
        Returns a list of (filename_stem, target_value, sample_id, species) for each row in the CSV.
        """
        data_list = []
        for _, row in self.df.iterrows():
            filename_stem = row["file"]  # or whatever your CSV column is called
            # remove suffix if it exists
            filename_stem = filename_stem.split(".")[0]
            target_value = (
                float(row[self.target_col])  # e.g. "age"
                if self.target_col in self.df.columns
                else float("nan")  # hidden test set: no reference
            )
            sample_id = row["sampleID"]
            species = str(row["species"]) if "species" in self.df.columns else ""
            data_list.append((filename_stem, target_value, sample_id, species))
        return data_list

    def _load_data(self, idx):
        """
        Load the raw point cloud from disk, do optional farthest-point sampling.
        Returns a dictionary with keys: coord, (optionally normal), and regression.
        """
        data_name, target_value = self.data_list[idx][:2]
        if data_name in self.data_cache:
            return self.data_cache[data_name]

        # Path to the point cloud file
        pc_path = os.path.join(self.data_root, self.pc_folder, data_name,  'coord.npy')

        # Load the point cloud (adapt for your file format)
        # Example: .txt with Nx3 or Nx6 columns
        # data = np.loadtxt(pc_path, delimiter=",").astype(np.float32)
        data = np.load(pc_path).astype(np.float32)
        # print(data.shape)
        # If your file is Nx3 (just coordinates):
        coord = data[:, :3]
        normal = None
        # If Nx6: then
        # coord, normal = data[:, 0:3], data[:, 3:6]

        # Optionally sample points
        if self.num_point is not None and len(coord) > self.num_point:
            if self.uniform_sampling:
                # Use farthest point sampling
                with torch.no_grad():
                    # If you want to sample coords only, do:
                    points_torch = torch.tensor(coord).float()  # Nx3
                    mask = pointops.farthest_point_sampling(
                        points_torch.contiguous().cuda(),
                        torch.tensor([len(coord)]).long().cuda(),
                        torch.tensor([self.num_point]).long().cuda(),
                    )
                coord = coord[mask.cpu()]
                if normal is not None:
                    normal = normal[mask.cpu()]
            else:
                # Just slice the first num_point
                coord = coord[: self.num_point]
                if normal is not None:
                    normal = normal[: self.num_point]

        data_dict = {
            "coord": coord,
            "regression": np.array([target_value], dtype=np.float32),
        }
        if normal is not None:
            data_dict["normal"] = normal

        return data_dict

    def get_data_name(self, idx):
        """
        Returns the 'filename_stem' used for file paths and caching.
        """
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx][0]  # (filename_stem, target_value, sample_id)

    def get_sample_id(self, idx):
        """
        Returns the 'sampleID' from the CSV, used as the submission identifier.
        """
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx][2]  # (filename_stem, target_value, sample_id, species)

    def get_species(self, idx):
        """
        Returns the 'species' label from the CSV (e.g. 'pine'/'spruce'), used for plot colouring.
        """
        data_idx = idx % len(self.data_list)
        return self.data_list[data_idx][3]  # (filename_stem, target_value, sample_id, species)

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        elif self.inference_mode:
            return self.prepare_inference_data(idx)
        else:
            return self.prepare_train_data(idx)

    def prepare_train_data(self, idx):
        data_dict = self._get_item_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_inference_data(self, idx):
        data_dict = self._get_item_data(idx)
        data_dict = self.transform(data_dict)
        data_dict["data_name"] = self.get_sample_id(idx)
        data_dict["species"] = self.get_species(idx)
        return data_dict

    def prepare_test_data(self, idx):
        """
        Example test-time augmentation approach (like ModelNetDataset).
        If no test_cfg, just do the same as inference or train.
        """
        if not self.test_cfg:
            return self.prepare_inference_data(idx)

        data_dict = self._get_item_data(idx)
        base_category = data_dict.pop("regression", None)  # store the numeric target

        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(aug(deepcopy(data_dict)))
        for i in range(len(data_dict_list)):
            data_dict_list[i] = self.post_transform(data_dict_list[i])

        data_dict = dict(
            voting_list=data_dict_list,
            regression=base_category,
            name=self.get_sample_id(idx),
        )
        return data_dict

    def _get_item_data(self, idx):
        """
        Retrieve from cache or load raw from disk.
        """
        data_idx = idx % len(self.data_list)
        data_name = self.data_list[data_idx][0]
        if data_name in self.data_cache:
            data_dict = self.data_cache[data_name]
        else:
            data_dict = self._load_data(data_idx)
        return deepcopy(data_dict)
