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
    shift = tl.maximum(maximum, -1.0e30)
    value = shift + tl.log(tl.sum(tl.exp(terms - shift), axis=0) + tl.exp(unpaired - shift))
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
    shift = tl.maximum(maximum, -1.0e30)
    value = shift + tl.log(tl.sum(tl.exp(terms - shift), axis=0))
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
    shift = tl.maximum(tl.maximum(old, update), -1.0e30)
    update = shift + tl.log(tl.exp(old - shift) + tl.exp(update - shift))
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


@triton.jit
def _inside_tangent_span(scores, direction, mask, z, dz, length, span,
                         block: tl.constexpr, min_loop: tl.constexpr):
    """Directional derivative of the exact inside recurrence."""
    i = tl.program_id(0)
    j = i + span
    m = tl.arange(0, block)
    safe_m = tl.minimum(m, span - min_loop - 1)
    left_index = i * length + i + m - 1
    left = tl.load(z + tl.maximum(left_index, 0), mask=m >= 1, other=0.0)
    dleft = tl.load(dz + tl.maximum(left_index, 0), mask=m >= 1, other=0.0)
    middle_index = (i + safe_m + 1) * length + j - 1
    middle = tl.load(z + middle_index, mask=m < span - min_loop, other=0.0)
    dmiddle = tl.load(dz + middle_index, mask=m < span - min_loop, other=0.0)
    pair_index = (i + safe_m) * length + j
    legal = tl.load(mask + pair_index, mask=m < span - min_loop, other=0).to(tl.int1)
    score = tl.load(scores + pair_index, mask=m < span - min_loop, other=0.0)
    dscore = tl.load(direction + pair_index, mask=m < span - min_loop, other=0.0)
    active = legal & (m < span - min_loop)
    terms = tl.where(active, left + middle + score, -float("inf"))
    dterms = tl.where(active, dleft + dmiddle + dscore, 0.0)
    unpaired = tl.load(z + i * length + j - 1)
    dunpaired = tl.load(dz + i * length + j - 1)
    value = tl.load(z + i * length + j)
    finite_value = value != -float("inf")
    weighted = tl.where(active & finite_value, tl.exp(terms - value) * dterms, 0.0)
    dvalue = (tl.sum(weighted, axis=0)
              + tl.where(finite_value, tl.exp(unpaired - value) * dunpaired, 0.0))
    tl.store(dz + i * length + j, dvalue)


@triton.jit
def _outside_forward_tangent_span(scores, direction, mask, z, dz, outside, parents,
                                  doutside, dparents, length, span,
                                  block: tl.constexpr, min_loop: tl.constexpr):
    """Recompute the outside forward and its directional derivative together.

    Keeping parent forward/tangent states in the same recurrence avoids storing
    O(L^3) intermediate parent tables solely for automatic differentiation.
    """
    i = tl.program_id(0)
    j = i + span
    left = tl.load(z + tl.maximum(i - 1, 0), mask=i >= 1, other=0.0)
    dleft = tl.load(dz + tl.maximum(i - 1, 0), mask=i >= 1, other=0.0)
    right_index = tl.minimum(j + 1, length - 1) * length + length - 1
    right = tl.load(z + right_index, mask=j + 1 < length, other=0.0)
    dright = tl.load(dz + right_index, mask=j + 1 < length, other=0.0)
    first = left + right
    dfirst = dleft + dright
    t = tl.arange(0, block)
    safe_t = tl.minimum(t, length - 1)
    valid_t = (t >= j + 1) & (t < length)
    parent = tl.load(parents + i * length + safe_t, mask=valid_t, other=-float("inf"))
    dparent = tl.load(dparents + i * length + safe_t, mask=valid_t, other=0.0)
    middle_index = (j + 1) * length + tl.maximum(safe_t - 1, 0)
    middle = tl.load(z + middle_index, mask=valid_t, other=0.0)
    dmiddle = tl.load(dz + middle_index, mask=valid_t, other=0.0)
    parent_terms = tl.where(valid_t, parent + middle, -float("inf"))
    dparent_terms = tl.where(valid_t, dparent + dmiddle, 0.0)
    terms = tl.where(t == 0, first, parent_terms)
    dterms = tl.where(t == 0, dfirst, dparent_terms)
    maximum = tl.max(terms, axis=0)
    shift = tl.maximum(maximum, -1.0e30)
    value = shift + tl.log(tl.sum(tl.exp(terms - shift), axis=0))
    finite_value = value != -float("inf")
    dvalue = tl.sum(tl.where(finite_value, tl.exp(terms - value) * dterms, 0.0), axis=0)
    pair_index = i * length + j
    score = tl.load(scores + pair_index)
    dscore = tl.load(direction + pair_index)
    legal = tl.load(mask + pair_index).to(tl.int1)
    tl.store(outside + pair_index, value)
    tl.store(doutside + pair_index, dvalue)
    m = tl.arange(0, block)
    child = i + m
    safe_child = tl.minimum(child, length - 1)
    zg_index = (i + 1) * length + tl.maximum(child - 1, 0)
    zg = tl.load(z + zg_index, mask=(m >= 1) & (child < j), other=0.0)
    dzg = tl.load(dz + zg_index, mask=(m >= 1) & (child < j), other=0.0)
    update = value + score + zg
    dupdate = dvalue + dscore + dzg
    old = tl.load(parents + safe_child * length + j,
                  mask=(m >= 1) & (child < j), other=-float("inf"))
    dold = tl.load(dparents + safe_child * length + j,
                   mask=(m >= 1) & (child < j), other=0.0)
    parent_shift = tl.maximum(tl.maximum(old, update), -1.0e30)
    new = parent_shift + tl.log(tl.exp(old - parent_shift) + tl.exp(update - parent_shift))
    finite_new = new != -float("inf")
    old_weight = tl.where(finite_new, tl.exp(old - new), 0.0)
    update_weight = tl.where(finite_new, tl.exp(update - new), 0.0)
    dnew = old_weight * dold + update_weight * dupdate
    active = legal & (m >= 1) & (child < j)
    tl.store(parents + safe_child * length + j, new, mask=active)
    tl.store(dparents + safe_child * length + j, dnew, mask=active)


@triton.jit
def _marginal_tangent(scores, direction, mask, z, dz, outside, doutside,
                      probabilities, length, block: tl.constexpr,
                      min_loop: tl.constexpr):
    i = tl.program_id(0)
    j = tl.arange(0, block)
    safe_j = tl.minimum(j, length - 1)
    valid = (j > i + min_loop) & (j < length)
    pair_index = i * length + safe_j
    legal = tl.load(mask + pair_index, mask=valid, other=0).to(tl.int1)
    score = tl.load(scores + pair_index, mask=valid, other=0.0)
    dscore = tl.load(direction + pair_index, mask=valid, other=0.0)
    o = tl.load(outside + pair_index, mask=valid, other=-float("inf"))
    do = tl.load(doutside + pair_index, mask=valid, other=0.0)
    zg_index = (i + 1) * length + safe_j - 1
    zg = tl.load(z + zg_index, mask=valid, other=0.0)
    dzg = tl.load(dz + zg_index, mask=valid, other=0.0)
    log_partition = tl.load(z + length - 1)
    dlog_partition = tl.load(dz + length - 1)
    probability = tl.exp(o + score + zg - log_partition)
    probability = tl.minimum(tl.maximum(probability, 0.0), 1.0)
    tangent = probability * (do + dscore + dzg - dlog_partition)
    tangent = tl.where(legal & valid, tangent, 0.0)
    tl.store(probabilities + pair_index, tangent, mask=valid)


def cuda_marginal_tangent(scores, mask, direction):
    """Exact J_marginals @ direction in float64 on the production CUDA path."""
    if not scores.is_cuda or not mask.is_cuda or not direction.is_cuda:
        raise ValueError("CUDA marginal tangent requires CUDA tensors")
    if scores.shape != mask.shape or scores.shape != direction.shape:
        raise ValueError("scores, mask and direction must have matching shapes")
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("marginal tangent requires square matrices")
    length = scores.shape[0]
    scores = scores.detach().to(torch.float64).contiguous()
    direction = direction.detach().to(torch.float64).contiguous()
    mask = mask.detach().to(torch.bool).contiguous()
    z = torch.zeros((length, length), dtype=torch.float64, device=scores.device)
    dz = torch.zeros_like(z)
    tangent = torch.zeros_like(z)
    if length < 5:
        return tangent
    block = max(2, triton.next_power_of_2(length))
    for span in range(4, length):
        _inside_span[(length - span,)](scores, mask, z, length, span,
                                      block=block, min_loop=3, num_warps=4)
    for span in range(4, length):
        _inside_tangent_span[(length - span,)](
            scores, direction, mask, z, dz, length, span,
            block=block, min_loop=3, num_warps=4)
    outside = torch.full_like(z, -float("inf"))
    parents = torch.full_like(z, -float("inf"))
    doutside = torch.zeros_like(z)
    dparents = torch.zeros_like(z)
    for span in range(length - 1, 3, -1):
        _outside_forward_tangent_span[(length - span,)](
            scores, direction, mask, z, dz, outside, parents, doutside, dparents,
            length, span, block=block, min_loop=3, num_warps=4)
    _marginal_tangent[(length,)](scores, direction, mask, z, dz, outside,
                                 doutside, tangent, length, block=block,
                                 min_loop=3, num_warps=4)
    return tangent


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
