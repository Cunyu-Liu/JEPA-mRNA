import numpy as np
import torch

from tools.dp_decode_c_wrapper import nussinov_map_batch
from kirigami.utils import PSEUDO_LEFT, PSEUDO_RIGHT


def dict2db(idxs, length):
    pairs = {(min(i, j), max(i, j)) for i, j in idxs if i != j}
    pair_hierarch = []
    mask = np.triu(np.ones((length, length), dtype=np.int8), k=1)
    while pairs:
        scores = np.zeros((length, length), dtype=np.float64)
        for i, j in pairs:
            scores[i, j] = scores[j, i] = 1.0
        nested = set(map(tuple, nussinov_map_batch(scores[None, :, :], mask[None, :, :])[0]))
        if not nested:
            raise RuntimeError("C decoder returned no pairs for a nonempty input")
        pairs.difference_update(nested)
        pair_hierarch.append(nested)
    out_list = length * ["."]
    for left, right, pair_set in zip(PSEUDO_LEFT, PSEUDO_RIGHT, pair_hierarch):
        for pair in pair_set:
            i, j = min(pair), max(pair)
            out_list[i] = left
            out_list[j] = right
    return "".join(out_list)


def mat2db(mat):
    mat_ = mat.squeeze()
    length = mat_.shape[0]
    idxi, idxj = torch.where(mat_)
    return dict2db(list(zip(idxi.tolist(), idxj.tolist())), length)
