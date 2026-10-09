import os
import warnings
from collections import abc
import numpy as np
import torch
from importlib import import_module
from torch.nn.functional import normalize, softmax

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def intersection_and_union(output, target, K, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersection_and_union_gpu(output, target, k, ignore_index=-1):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=k, min=0, max=k - 1)
    area_output = torch.histc(output, bins=k, min=0, max=k - 1)
    area_target = torch.histc(target, bins=k, min=0, max=k - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def make_dirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Default: False.

    Returns:
        list[module] | module | None: The imported modules.

    Examples:
        >>> osp, sys = import_modules_from_strings(
        ...     ['os.path', 'sys'])
        >>> import os.path as osp_
        >>> import sys as sys_
        >>> assert osp == osp_
        >>> assert sys == sys_
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(f"custom_imports must be a list but got type {type(imports)}")
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(f"{imp} is of type {type(imp)} and cannot be imported.")
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f"{imp} failed to import and is ignored.", UserWarning)
                imported_tmp = None
            else:
                raise ImportError
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


class DummyClass:
    def __init__(self):
        pass



### similarity of all possible pairs of elements in a tensor
def measure_similarity_full(features, indexes, normalized=True):
    selected_features = features[indexes]
    if not normalized:
        selected_features = normalize(selected_features, dim=1, p=2)
    similarity = torch.mm(selected_features, selected_features.T)
    num_pairs = (len(selected_features)**2 - len(selected_features)) / 2 
    similarity_avg = torch.triu(similarity, diagonal=1).sum() / num_pairs
    return similarity, similarity_avg

### similarity of pairs of elements in a tensor given source and target indexes
def measure_similarity_partial(features, s, t, normalized=True):
    if not normalized:
        features = normalize(features, dim=1, p=2)
    q = features[s]
    k = features[t]
    similarity = torch.einsum('ed, ed -> e', q, k)
    similarity_avg = similarity.mean()
    return similarity, similarity_avg

### similarity of pairs of elements in two tensors (row-wise)
def measure_similarity(features1, features2, normalized=True):
    if not normalized:
        features1 = normalize(features1, dim=1, p=2)
        features2 = normalize(features2, dim=1, p=2)

    similarity = torch.einsum('ed, ed -> e', features1, features2)
    similarity_avg = similarity.mean()
    return similarity, similarity_avg



def knn_segmentor(feat, train_features, train_labels_hist, num_classes, T, nb_knn):
    similarity = torch.mm(feat, train_features.T)
    topk_sims, indices = similarity.topk(200, largest=True, sorted=True)
    candidate_labels = train_labels_hist.unsqueeze(0).expand(len(similarity), -1,-1)
    neighbors_labels = torch.gather(candidate_labels, 1, indices.unsqueeze(-1).expand(-1,-1,num_classes))
    
    batch_size = neighbors_labels.shape[0]
    topk_sims_transform = softmax(topk_sims / T, 1)
    matmul = torch.mul(
        # one_hot(neighbors_labels, num_classes=num_classes),  # [20043, 200, 20]
        neighbors_labels,
        topk_sims_transform.view(batch_size, -1, 1), # [20043, 200, 1]
    )
    probas_for_k = {k: torch.sum(matmul[:, :k, :], 1) for k in nb_knn}

    del matmul
    del similarity
    return probas_for_k