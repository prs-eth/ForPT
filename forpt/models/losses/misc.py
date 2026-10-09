import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from forpt.utils.logger import get_root_logger
from .builder import LOSSES


# Cache keyed on the source spec: several criteria in one run would otherwise
# re-read the same CSV, and the table is identical for identical specs.
_AGE_WEIGHT_CACHE = {}


def build_age_weight_table(
    data_root="data/forage",
    metadata_file="FORage_tree_metadata_train_val.csv",
    split="train",
    pc_folder="pc",
    target_col="age",
    sample_list_file=None,
    match_full_split_scale=True,
):
    """Per-age inverse-frequency weights, computed from the training set in use.

    weight[a] = (1 / count_a) / sum_a'(1 / count_a'), so the weights sum to 1
    over the distinct ages actually present -- the same normalisation the old
    hardcoded `train_loss_weights` table used, but derived from the data instead
    of frozen at the 100% split. `sample_list_file` must mirror the one given to
    the train dataset, otherwise a data-efficiency run reweights by frequencies
    its subset does not have.

    The row filtering mirrors ForAgeDataset: split column, then "the point
    cloud exists on disk", then the optional subset list.

    Ages never seen in training inherit the weight of the nearest observed age
    rather than a fixed 1.0, which would otherwise dominate any loss computed
    over val/test. The table spans the largest age in the whole CSV so that
    evaluating on a split with older trees than the training subset cannot run
    off the end of it.

    match_full_split_scale rescales a subset table so a training sample gets the
    same weight on average as it would under the full split. Only the relative
    emphasis across ages should follow the subset -- left alone, the 1% loss
    comes out ~46x the 100% loss and the data-efficiency curve confounds the
    reweighting with an effective learning-rate change. Equivalent to scaling
    `loss_weight` per subset, but computed from the data instead of hardcoded.
    A no-op when sample_list_file is None.
    """
    key = (
        data_root,
        metadata_file,
        split,
        pc_folder,
        target_col,
        sample_list_file,
        match_full_split_scale,
    )
    if key in _AGE_WEIGHT_CACHE:
        return _AGE_WEIGHT_CACHE[key].clone()

    logger = get_root_logger()
    metadata_path = os.path.join(data_root, metadata_file)
    if not os.path.isfile(metadata_path):
        logger.warning(
            f"age weights: {metadata_path} not found, falling back to the hardcoded "
            f"train_loss_weights table (valid only for the 100% train split)"
        )
        observed = np.array(sorted(train_loss_weights), dtype=np.int64)
        weights = np.array([train_loss_weights[a] for a in observed], dtype=np.float64)
        table = torch.from_numpy(
            _fill_nearest(observed, weights, int(observed.max()))
        ).float()
        _AGE_WEIGHT_CACHE[key] = table
        return table.clone()

    df = pd.read_csv(metadata_path)
    max_age = int(df[target_col].max())

    if "split" in df.columns:
        df = df[df["split"] == split]
    pc_dir = os.path.join(data_root, pc_folder)
    if os.path.isdir(pc_dir):
        valid_files = {name + ".laz" for name in os.listdir(pc_dir)}
        df = df[df["file"].isin(valid_files)]
    full_split_ages = df[target_col].to_numpy().astype(np.int64)

    if sample_list_file is not None:
        subset = set(np.atleast_1d(np.loadtxt(sample_list_file, dtype=str)).tolist())
        df = df[df["file"].isin(subset)]

    ages = df[target_col].to_numpy().astype(np.int64)
    assert len(ages) > 0, f"no samples left to compute age weights from ({key})"

    observed, counts, unscaled = _inverse_frequency_weights(ages)
    weights, scale = unscaled, 1.0
    if match_full_split_scale and sample_list_file is not None:
        _, full_counts, full_weights = _inverse_frequency_weights(full_split_ages)
        scale = _mean_per_sample(full_counts, full_weights) / _mean_per_sample(
            counts, unscaled
        )
        weights = unscaled * scale

    table = _fill_nearest(observed, weights, max(max_age, int(observed.max())))
    subset_info = (
        f", subset {os.path.basename(sample_list_file)}"
        if sample_list_file is not None
        else ""
    )
    scale_info = f", rescaled by {scale:.6f}" if scale != 1.0 else ""
    logger.info(
        f"age weights: {len(ages)} {split} samples{subset_info}, "
        f"{len(observed)} distinct ages, weight range "
        f"[{weights.min():.8f}, {weights.max():.8f}], "
        f"mean per sample {_mean_per_sample(counts, weights):.8f}{scale_info}"
    )
    _log_age_weights(
        logger, observed, counts, weights, unscaled if scale != 1.0 else None
    )

    table = torch.from_numpy(table).float()
    _AGE_WEIGHT_CACHE[key] = table
    return table.clone()


def _log_age_weights(logger, observed, counts, weights, unscaled=None):
    """Dump the whole age -> count -> weight table, so a run's log records the
    exact reweighting it used instead of just its summary statistics.

    `unscaled` is the inverse-frequency weight before match_full_split_scale;
    pass it to show both columns, so the log separates what the subset's own
    frequencies asked for from the constant that puts it on the full split's
    loss scale.
    """
    if unscaled is None:
        header = "age weights (age: count -> weight):"
        cells = [
            f"{age:>4}:{count:>5} -> {weight:.8f}"
            for age, count, weight in zip(observed, counts, weights)
        ]
        per_line = 4
    else:
        header = "age weights (age: count -> unscaled -> rescaled):"
        cells = [
            f"{age:>4}:{count:>5} -> {raw:.8f} -> {weight:.8f}"
            for age, count, raw, weight in zip(observed, counts, unscaled, weights)
        ]
        per_line = 3

    logger.info(header)
    for start in range(0, len(cells), per_line):
        logger.info("  " + "   ".join(cells[start : start + per_line]))


def _inverse_frequency_weights(ages):
    """Distinct ages, their counts, and weights summing to 1 over those ages."""
    observed, counts = np.unique(ages, return_counts=True)
    inv = 1.0 / counts
    return observed, counts, inv / inv.sum()


def _mean_per_sample(counts, weights):
    """Average weight a training sample receives, i.e. the loss's overall scale."""
    return float((counts * weights).sum() / counts.sum())


def _fill_nearest(observed, weights, max_age):
    """Expand per-age weights to a dense [0, max_age] lookup by nearest age."""
    grid = np.arange(max_age + 1)
    nearest = np.abs(grid[:, None] - observed[None, :]).argmin(axis=1)
    return weights[nearest]


def _register_age_weight_table(module, age_weights):
    """Attach the lookup table used by the *Weighted regression losses.

    Non-persistent: the table is derived from the dataset, so it has no place in
    a checkpoint, and keeping it out means a run resumed with a different subset
    rebuilds it instead of restoring a stale one.
    """
    table = build_age_weight_table(**(age_weights or {}))
    module.register_buffer("weight_table", table, persistent=False)


def _lookup_age_weights(module, target):
    index = target.long().clamp(0, module.weight_table.numel() - 1)
    return module.weight_table.to(index.device)[index]


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


@LOSSES.register_module()
class HuberLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        beta=1.0,
        label_smoothing=0.0,
        loss_weight=1.0,
    ):
        super(HuberLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.SmoothL1Loss(reduction=reduction, beta=beta)

    def forward(self, pred, target):
        return self.loss(pred, target) * self.loss_weight


# Legacy table: inverse-frequency weights frozen at the 100% train split of
# data/forage/FORage_tree_metadata_train_val.csv, rounded to 4 decimals. Superseded by
# build_age_weight_table(), which recomputes the same quantity from the training
# set actually in use; kept only as the fallback for when that CSV is missing.
train_loss_weights = {
    1: 0.0012, 2: 0.002, 3: 0.0008, 4: 0.0005, 5: 0.0006, 6: 0.0006, 7: 0.0007, 
    8: 0.0006, 9: 0.0009, 10: 0.0012, 11: 0.0013, 12: 0.0012, 13: 0.0016, 14: 0.0024, 
    15: 0.0024, 16: 0.0014, 17: 0.0018, 18: 0.0028, 19: 0.0014, 20: 0.0009, 21: 0.0007, 
    22: 0.0006, 23: 0.0008, 24: 0.0008, 25: 0.0009, 26: 0.0014, 27: 0.0012, 28: 0.0012, 
    29: 0.0008, 30: 0.0007, 31: 0.0009, 32: 0.0009, 33: 0.0009, 34: 0.001, 35: 0.0011, 
    36: 0.001, 37: 0.0014, 38: 0.0009, 39: 0.0007, 40: 0.0007, 41: 0.0008, 42: 0.0004, 
    43: 0.0005, 44: 0.0009, 45: 0.0006, 46: 0.0008, 47: 0.0007, 48: 0.0005, 49: 0.001, 
    50: 0.0009, 51: 0.0014, 52: 0.0007, 53: 0.0011, 54: 0.0014, 55: 0.0016, 56: 0.0024, 
    57: 0.0013, 58: 0.0011, 59: 0.0035, 60: 0.0007, 61: 0.0024, 62: 0.0009, 63: 0.0011, 
    64: 0.0024, 65: 0.0035, 67: 0.0005, 68: 0.0035, 69: 0.0047, 70: 0.0018, 71: 0.0071, 
    72: 0.0035, 73: 0.0071, 74: 0.0047, 75: 0.0035, 76: 0.0141, 78: 0.0071, 80: 0.0071, 
    81: 0.0071, 84: 0.0035, 85: 0.0035, 86: 0.0071, 87: 0.0028, 88: 0.0028, 89: 0.0028, 
    90: 0.0047, 91: 0.0035, 92: 0.0047, 93: 0.002, 94: 0.0028, 95: 0.0035, 96: 0.0047, 
    97: 0.0047, 98: 0.0141, 99: 0.0047, 100: 0.0028, 101: 0.0035, 102: 0.0035, 104: 0.0035, 
    106: 0.0035, 108: 0.0047, 109: 0.0024, 112: 0.0047, 115: 0.0141, 116: 0.0071, 117: 0.0047, 
    119: 0.0071, 120: 0.0141, 123: 0.0071, 124: 0.0035, 126: 0.0071, 127: 0.0141, 128: 0.0141, 
    130: 0.0047, 133: 0.0141, 138: 0.0071, 139: 0.0035, 140: 0.0028, 141: 0.0141, 142: 0.0024, 
    143: 0.0141, 145: 0.0071, 146: 0.0047, 147: 0.0071, 149: 0.0141, 150: 0.0141, 152: 0.0141, 
    153: 0.0047, 155: 0.0141, 156: 0.002, 157: 0.0028, 160: 0.0071, 162: 0.0141, 163: 0.0071, 
    168: 0.0071, 169: 0.0141, 171: 0.0071, 172: 0.0141, 173: 0.0141, 174: 0.0071, 175: 0.0071, 
    176: 0.0071, 179: 0.0141, 180: 0.0071, 182: 0.0141, 186: 0.0141, 187: 0.0141, 188: 0.0141, 
    199: 0.0141, 201: 0.0141, 202: 0.0071, 203: 0.0141, 206: 0.0141, 208: 0.0141, 209: 0.0047, 
    210: 0.0141, 211: 0.0047, 212: 0.0071, 214: 0.0071, 215: 0.0047, 216: 0.0141, 219: 0.0141, 
    220: 0.0071, 221: 0.0141, 222: 0.0141, 223: 0.0141, 225: 0.0141, 255: 0.0141, 258: 0.0141, 
    272: 0.0141, 289: 0.0141, 291: 0.0141, 313: 0.0141, 334: 0.0141, 348: 0.0141}

    

@LOSSES.register_module()
class MSELossWeighted(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean", age_weights=None):
        """
        Mean Squared Error Loss (MSE Loss) with weighting based on the target value.
        Args:
            loss_weight (float): Multiplicative factor applied to the final loss.
            reduction (str): Specifies the reduction to apply: "mean", "sum", or "none".
            age_weights (dict): Overrides for build_age_weight_table. A subset run
                must pass the same `sample_list_file` as its train dataset.
        """
        super(MSELossWeighted, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.base_loss = nn.MSELoss(reduction="none")

        _register_age_weight_table(self, age_weights)

    def forward(self, pred, target):
        """
        Computes the weighted MSE loss.
        Args:
            pred (torch.Tensor): The predictions.
            target (torch.Tensor): The ground truth targets. Expected to be integer ages.
        Returns:
            torch.Tensor: The computed weighted loss.
        """
        loss = self.base_loss(pred, target)
        weights = _lookup_age_weights(self, target)
        weighted_loss = loss * weights
        
        if self.reduction == "mean":
            final_loss = weighted_loss.mean()
        elif self.reduction == "sum":
            final_loss = weighted_loss.sum()
        else:
            final_loss = weighted_loss
        
        return final_loss * self.loss_weight
