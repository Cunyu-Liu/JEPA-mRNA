from rnajepa.harness import nussinov_map
from solver_ablation import crossing_rate, greedy_pairs


def decode_scores(scores, mask, decoder):
    if decoder == "nussinov":
        return [tuple(pair) for pair in nussinov_map(scores, mask)], 0, 0
    if decoder != "greedy":
        raise ValueError(decoder)
    pairs = [tuple(pair) for pair in greedy_pairs(scores, mask)]
    used = set()
    conflicts = 0
    for i, j in pairs:
        if i in used or j in used:
            conflicts += 1
        used.update((i, j))
    return pairs, crossing_rate(pairs), conflicts
