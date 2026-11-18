"""Local-to-global RNA secondary-structure metrics for the FoldNAR E0 gate.

Coordinates are zero-based, inclusive, and pairs are canonical ``(i, j), i < j``.
The structural metrics operate on nested, one-pair-per-nucleotide structures.
Crossing or otherwise invalid inputs are rejected explicitly instead of being
silently repaired.  A helix/loop tree contains one node per maximal stem; its
label is the kind of loop closed by the stem's outer pair.  The artificial root
is the external loop.  Topology GED is unit-cost ordered, labelled tree edit
distance on this representation, not base-pair edit distance.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

Pair = tuple[int, int]
LOOP_KINDS = ("EXTERNAL", "HAIRPIN", "INTERNAL", "BULGE", "MULTILOOP")


class StructureError(ValueError):
    """Raised when a pair set has no unambiguous nested loop/helix tree."""


def canonical_pairs(pairs: Iterable[Sequence[int]], length: Optional[int] = None) -> frozenset[Pair]:
    out: set[Pair] = set()
    used: set[int] = set()
    for raw in pairs:
        if len(raw) != 2:
            raise StructureError(f"pair does not have two coordinates: {raw!r}")
        i, j = sorted((int(raw[0]), int(raw[1])))
        if i == j:
            raise StructureError(f"self pair: {(i, j)}")
        if i < 0 or (length is not None and j >= length):
            raise StructureError(f"out-of-range pair {(i, j)} for length {length}")
        if (i, j) in out:
            raise StructureError(f"duplicate pair {(i, j)}")
        if i in used or j in used:
            raise StructureError(f"nucleotide used more than once at {(i, j)}")
        out.add((i, j))
        used.add(i)
        used.add(j)
    ordered = sorted(out)
    for k, (i, j) in enumerate(ordered):
        for l, m in ordered[k + 1 :]:
            if i < l < j < m:
                raise StructureError(f"crossing pairs {(i, j)} and {(l, m)}")
            if l >= j:
                break
    return frozenset(out)


def pair_prf(pred: Iterable[Sequence[int]], truth: Iterable[Sequence[int]]) -> dict[str, float | int]:
    p = {tuple(sorted(map(int, x))) for x in pred}
    g = {tuple(sorted(map(int, x))) for x in truth}
    tp, fp, fn = len(p & g), len(p - g), len(g - p)
    return {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else (1.0 if fp == 0 else 0.0)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    return {"precision": precision, "recall": recall, "f1": f1}


def extract_stems(pairs: Iterable[Sequence[int]]) -> tuple[tuple[Pair, ...], ...]:
    """Return maximal consecutive nested stems, outer pair first, no bulges."""
    pset = set(canonical_pairs(pairs))
    stems: list[tuple[Pair, ...]] = []
    for i, j in sorted(pset):
        if (i - 1, j + 1) in pset:
            continue
        stem: list[Pair] = []
        a, b = i, j
        while (a, b) in pset:
            stem.append((a, b))
            a, b = a + 1, b - 1
        stems.append(tuple(stem))
    if sum(map(len, stems)) != len(pset):
        raise AssertionError("stem partition lost pairs")
    return tuple(stems)


@dataclass(frozen=True)
class Loop:
    closing: Optional[Pair]
    children: tuple[Pair, ...]
    kind: str

    @property
    def identity(self) -> tuple:
        return (self.closing, self.children)


@dataclass(frozen=True)
class HelixNode:
    stem: tuple[Pair, ...]
    loop_kind: str
    children: tuple["HelixNode", ...] = ()

    @property
    def outer_pair(self) -> Pair:
        return self.stem[0]


def pair_children(pairs: Iterable[Sequence[int]]) -> tuple[dict[Optional[Pair], tuple[Pair, ...]], dict[Pair, Optional[Pair]]]:
    """Build the nesting tree with a sequence-ordered sweep."""
    pset = canonical_pairs(pairs)
    mutable: dict[Optional[Pair], list[Pair]] = {None: []}
    parent: dict[Pair, Optional[Pair]] = {}
    stack: list[Pair] = []
    for pair in sorted(pset, key=lambda x: (x[0], -x[1])):
        i, j = pair
        while stack and not (stack[-1][0] < i and j < stack[-1][1]):
            stack.pop()
        par = stack[-1] if stack else None
        parent[pair] = par
        mutable.setdefault(par, []).append(pair)
        mutable.setdefault(pair, [])
        stack.append(pair)
    children = {key: tuple(sorted(value)) for key, value in mutable.items()}
    return children, parent


def decompose_loops(pairs: Iterable[Sequence[int]]) -> tuple[Loop, ...]:
    """Decompose loops between maximal stems, not between adjacent stem pairs."""
    pset = canonical_pairs(pairs)
    stems = extract_stems(pset)
    nesting, parent = pair_children(pset)
    del nesting  # only the parent relation is needed below
    pair_to_stem = {pair: index for index, stem in enumerate(stems) for pair in stem}
    child_stems: dict[Optional[Pair], list[Pair]] = {None: []}
    for stem in stems:
        outer = stem[0]
        parent_pair = parent[outer]
        parent_outer = stems[pair_to_stem[parent_pair]][0] if parent_pair is not None else None
        child_stems.setdefault(parent_outer, []).append(outer)
        child_stems.setdefault(outer, [])
    loops: list[Loop] = [Loop(None, tuple(sorted(child_stems[None])), "EXTERNAL")]
    for closing in sorted(key for key in child_stems if key is not None):
        nested = tuple(sorted(child_stems[closing]))
        if not nested:
            kind = "HAIRPIN"
        elif len(nested) >= 2:
            kind = "MULTILOOP"
        else:
            child = nested[0]
            left = child[0] - closing[0] - 1
            right = closing[1] - child[1] - 1
            kind = "INTERNAL" if left > 0 and right > 0 else "BULGE"
        loops.append(Loop(closing, nested, kind))
    return tuple(loops)


def build_helix_tree(pairs: Iterable[Sequence[int]]) -> tuple[HelixNode, ...]:
    pset = canonical_pairs(pairs)
    stems = extract_stems(pset)
    children, parent = pair_children(pset)
    pair_to_stem: dict[Pair, int] = {}
    for index, stem in enumerate(stems):
        for pair in stem:
            pair_to_stem[pair] = index
    loop_by_closing = {loop.closing: loop for loop in decompose_loops(pset)}
    child_indices: dict[Optional[int], list[int]] = {None: []}
    for index, stem in enumerate(stems):
        par_pair = parent[stem[0]]
        par_stem = pair_to_stem[par_pair] if par_pair is not None else None
        child_indices.setdefault(par_stem, []).append(index)
        child_indices.setdefault(index, [])

    def visit(index: int) -> HelixNode:
        stem = stems[index]
        return HelixNode(
            stem=stem,
            loop_kind=loop_by_closing[stem[0]].kind,
            children=tuple(visit(k) for k in sorted(child_indices[index], key=lambda k: stems[k][0])),
        )

    return tuple(visit(k) for k in sorted(child_indices[None], key=lambda k: stems[k][0]))


def topology_signature(roots: Sequence[HelixNode]) -> tuple:
    def encode(node: HelixNode) -> tuple:
        return (node.loop_kind, tuple(encode(child) for child in node.children))
    return ("EXTERNAL", tuple(encode(root) for root in roots))


def _flatten_postorder(roots: Sequence[HelixNode]) -> tuple[list[str], list[int], list[int]]:
    """Return labels, leftmost-leaf indices and Zhang-Shasha key roots."""
    labels: list[str] = []
    leftmost: list[int] = []

    def visit(node: HelixNode) -> int:
        child_leaves = [visit(child) for child in node.children]
        index = len(labels)
        labels.append(node.loop_kind)
        leaf = child_leaves[0] if child_leaves else index
        leftmost.append(leaf)
        return leaf

    # A shared external root makes the distance a tree distance, not a forest distance.
    synthetic = HelixNode((), "EXTERNAL", tuple(roots))
    visit(synthetic)
    keyroots: list[int] = []
    seen: set[int] = set()
    for index in range(len(labels) - 1, -1, -1):
        leaf = leftmost[index]
        if leaf not in seen:
            keyroots.append(index)
            seen.add(leaf)
    return labels, leftmost, sorted(keyroots)


def topology_ged(roots_a: Sequence[HelixNode], roots_b: Sequence[HelixNode]) -> int:
    """Zhang-Shasha ordered labelled tree edit distance, unit edit costs."""
    a, la, ka = _flatten_postorder(roots_a)
    b, lb, kb = _flatten_postorder(roots_b)
    distances: dict[tuple[int, int], int] = {}

    def solve(i: int, j: int) -> None:
        # Forest coordinates are local; node coordinates are one-based postorder.
        fd = [[0] * (j - lb[j] + 2) for _ in range(i - la[i] + 2)]
        for x in range(1, len(fd)):
            fd[x][0] = x
        for y in range(1, len(fd[0])):
            fd[0][y] = y
        for x in range(1, len(fd)):
            ai = la[i] + x - 1
            for y in range(1, len(fd[0])):
                bj = lb[j] + y - 1
                delete = fd[x - 1][y] + 1
                insert = fd[x][y - 1] + 1
                if la[ai] == la[i] and lb[bj] == lb[j]:
                    replace = fd[x - 1][y - 1] + (a[ai] != b[bj])
                    fd[x][y] = min(delete, insert, replace)
                    distances[(ai, bj)] = fd[x][y]
                else:
                    replace = fd[la[ai] - la[i]][lb[bj] - lb[j]] + distances[(ai, bj)]
                    fd[x][y] = min(delete, insert, replace)

    for i in ka:
        for j in kb:
            solve(i, j)
    return distances[(len(a) - 1, len(b) - 1)]


def _counter_prf(pred: Counter, truth: Counter) -> dict[str, float | int]:
    tp = sum((pred & truth).values())
    fp = sum((pred - truth).values())
    fn = sum((truth - pred).values())
    return {"tp": tp, "fp": fp, "fn": fn, **_prf(tp, fp, fn)}


def structure_metrics(pred_pairs: Iterable[Sequence[int]], truth_pairs: Iterable[Sequence[int]], length: Optional[int] = None) -> dict:
    pred = canonical_pairs(pred_pairs, length)
    truth = canonical_pairs(truth_pairs, length)
    pred_stems, truth_stems = extract_stems(pred), extract_stems(truth)
    pred_loops, truth_loops = decompose_loops(pred), decompose_loops(truth)
    pred_tree, truth_tree = build_helix_tree(pred), build_helix_tree(truth)
    pred_junctions = Counter(
        (loop.closing, child)
        for loop in pred_loops if loop.kind == "MULTILOOP" for child in loop.children
    )
    truth_junctions = Counter(
        (loop.closing, child)
        for loop in truth_loops if loop.kind == "MULTILOOP" for child in loop.children
    )
    ged = topology_ged(pred_tree, truth_tree)
    n_nodes = max(_tree_size(pred_tree), _tree_size(truth_tree))
    pair = pair_prf(pred, truth)
    return {
        "pair": pair,
        "stem": _counter_prf(Counter(pred_stems), Counter(truth_stems)),
        "loop": _counter_prf(Counter(x.identity for x in pred_loops), Counter(x.identity for x in truth_loops)),
        "loop_types": {
            kind: _counter_prf(
                Counter(x.identity for x in pred_loops if x.kind == kind),
                Counter(x.identity for x in truth_loops if x.kind == kind),
            ) for kind in LOOP_KINDS
        },
        "multiloop": _counter_prf(
            Counter(x.identity for x in pred_loops if x.kind == "MULTILOOP"),
            Counter(x.identity for x in truth_loops if x.kind == "MULTILOOP"),
        ),
        "junction": _counter_prf(pred_junctions, truth_junctions),
        "topology": {
            "ged": ged,
            "normalized_ged": ged / n_nodes if n_nodes else 0.0,
            "f1": 1.0 - ged / n_nodes if n_nodes else 1.0,
            "exact": topology_signature(pred_tree) == topology_signature(truth_tree),
            "n_pred_nodes": _tree_size(pred_tree),
            "n_truth_nodes": _tree_size(truth_tree),
        },
        "n_pred_pairs": len(pred),
        "n_truth_pairs": len(truth),
        "n_pred_stems": len(pred_stems),
        "n_truth_stems": len(truth_stems),
        "pred_loop_counts": dict(Counter(x.kind for x in pred_loops)),
        "truth_loop_counts": dict(Counter(x.kind for x in truth_loops)),
    }


def _tree_size(roots: Sequence[HelixNode]) -> int:
    return 1 + sum(1 + _tree_size_no_root(root.children) for root in roots)


def _tree_size_no_root(roots: Sequence[HelixNode]) -> int:
    return sum(1 + _tree_size_no_root(root.children) for root in roots)


def loop_diagram(pairs: Iterable[Sequence[int]], length: int) -> str:
    """Compact, lossless nesting diagram used in case-study JSON files."""
    pset = canonical_pairs(pairs, length)
    stems = extract_stems(pset)
    segments = []
    for stem in stems:
        outer, inner = stem[0], stem[-1]
        segments.append(f"{outer[0]}-{outer[1]}[stem={len(stem)};inner={inner[0]}-{inner[1]}]")
    loops = decompose_loops(pset)
    loop_text = ";".join(
        f"{x.kind}:{x.closing}->{list(x.children)}" for x in loops
    )
    return f"L={length} stems={' '.join(segments) or 'none'} loops={loop_text}"
