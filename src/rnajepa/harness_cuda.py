import torch
import triton
import triton.language as tl


@triton.jit
def _inside_span(scores, mask, z, length, span, block: tl.constexpr, min_loop: tl.constexpr):
    i = tl.program_id(0)
    j = i + span
    m = tl.arange(0, block)
    safe_m = tl.minimum(m, span - min_loop - 1)
    left_index = i * length + i + m - 1
    left = tl.load(z + tl.maximum(left_index, 0), mask=m >= 1, other=0.0)
    middle_index = (i + safe_m + 1) * length + j - 1
    middle = tl.load(z + middle_index, mask=m < span - min_loop, other=0.0)
    pair_index = (i + safe_m) * length + j
    legal = tl.load(mask + pair_index, mask=m < span - min_loop, other=0).to(tl.int1)
    score = tl.load(scores + pair_index, mask=m < span - min_loop, other=0.0)
    terms = tl.where(legal & (m < span - min_loop), left + middle + score, -float("inf"))
    unpaired = tl.load(z + i * length + j - 1)
    maximum = tl.maximum(unpaired, tl.max(terms, axis=0))
    value = maximum + tl.log(tl.sum(tl.exp(terms - maximum), axis=0) + tl.exp(unpaired - maximum))
    tl.store(z + i * length + j, value)


@triton.jit
def _outside_span(scores, mask, z, outside, parents, length, span,
                  block: tl.constexpr, min_loop: tl.constexpr):
    i = tl.program_id(0)
    j = i + span
    left = tl.load(z + tl.maximum(i - 1, 0), mask=i >= 1, other=0.0)
    right_index = tl.minimum(j + 1, length - 1) * length + length - 1
    right = tl.load(z + right_index, mask=j + 1 < length, other=0.0)
    first = left + right
    t = tl.arange(0, block)
    safe_t = tl.minimum(t, length - 1)
    valid_t = (t >= j + 1) & (t < length)
    parent = tl.load(parents + i * length + safe_t, mask=valid_t, other=-float("inf"))
    middle_index = (j + 1) * length + tl.maximum(safe_t - 1, 0)
    middle = tl.load(z + middle_index, mask=valid_t, other=0.0)
    parent_terms = tl.where(valid_t, parent + middle, -float("inf"))
    terms = tl.where(t == 0, first, parent_terms)
    maximum = tl.max(terms, axis=0)
    value = maximum + tl.log(tl.sum(tl.exp(terms - maximum), axis=0))
    pair_index = i * length + j
    score = tl.load(scores + pair_index)
    legal = tl.load(mask + pair_index).to(tl.int1)
    tl.store(outside + pair_index, value)
    m = tl.arange(0, block)
    child = i + m
    safe_child = tl.minimum(child, length - 1)
    zg_index = (i + 1) * length + tl.maximum(child - 1, 0)
    zg = tl.load(z + zg_index, mask=(m >= 1) & (child < j), other=0.0)
    update = value + score + zg
    old = tl.load(parents + safe_child * length + j, mask=(m >= 1) & (child < j), other=-float("inf"))
    update = tl.log(tl.exp(old) + tl.exp(update))
    tl.store(parents + safe_child * length + j, update,
             mask=legal & (m >= 1) & (child < j))


@triton.jit
def _marginals(scores, mask, z, outside, probabilities, log_z,
               length, block: tl.constexpr, min_loop: tl.constexpr):
    i = tl.program_id(0)
    j = tl.arange(0, block)
    safe_j = tl.minimum(j, length - 1)
    valid = (j > i + min_loop) & (j < length)
    pair_index = i * length + safe_j
    legal = tl.load(mask + pair_index, mask=valid, other=0).to(tl.int1)
    score = tl.load(scores + pair_index, mask=valid, other=0.0)
    o = tl.load(outside + pair_index, mask=valid, other=-float("inf"))
    zg = tl.load(z + (i + 1) * length + safe_j - 1, mask=valid, other=0.0)
    log_partition = tl.load(log_z)
    probability = tl.exp(o + score + zg - log_partition)
    probability = tl.minimum(tl.maximum(probability, 0.0), 1.0)
    probability = tl.where(legal & valid, probability, 0.0)
    tl.store(probabilities + pair_index, probability, mask=valid)
    tl.store(probabilities + safe_j * length + i, probability, mask=valid)


def cuda_inside_outside(scores, mask, min_loop=3):
    if not scores.is_cuda or not mask.is_cuda:
        raise ValueError("CUDA inside-outside requires CUDA scores and mask")
    if scores.ndim != 2 or scores.shape != mask.shape or scores.shape[0] != scores.shape[1]:
        raise ValueError("scores and mask must be matching square matrices")
    if min_loop != 3:
        raise ValueError("the production CUDA kernel requires min_loop=3")
    length = scores.shape[0]
    dtype = scores.dtype
    scores = scores.detach().to(torch.float64).contiguous()
    mask = mask.detach().to(torch.bool).contiguous()
    z = torch.zeros((length, length), dtype=torch.float64, device=scores.device)
    probabilities = torch.zeros_like(z)
    log_z = torch.zeros((), dtype=torch.float64, device=scores.device)
    if length == 0:
        return log_z.to(dtype), probabilities.to(dtype)
    block = max(2, triton.next_power_of_2(length))
    for span in range(min_loop + 1, length):
        _inside_span[(length - span,)](scores, mask, z, length, span,
                                      block=block, min_loop=min_loop, num_warps=4)
    log_z = z[0, length - 1]
    if length >= min_loop + 2:
        outside = torch.full_like(z, -float("inf"))
        parents = torch.full_like(z, -float("inf"))
        for span in range(length - 1, min_loop, -1):
            _outside_span[(length - span,)](scores, mask, z, outside, parents,
                                           length, span, block=block,
                                           min_loop=min_loop, num_warps=4)
        _marginals[(length,)](scores, mask, z, outside, probabilities, log_z,
                              length, block=block, min_loop=min_loop, num_warps=4)
    return log_z.to(dtype), probabilities.to(dtype)
