#!/usr/bin/env python3
"""C-accelerated batched DP decode (ctypes wrapper for dp_decode_c.so).

Drop-in replacement for the per-sequence nussinov_map() in harness.py,
but processes B sequences of the SAME length simultaneously with OpenMP.

Guarantee: bitwise-equal N/dec/kp tables to _max_dp_reference (verified by
test_bitwise_equal). The C recurrence and tie-breaking match the Python
reference exactly.
"""
import ctypes, os, sys
import numpy as np

_SO_PATH = os.path.join(os.path.dirname(__file__), 'dp_decode_c.so')
_lib = ctypes.CDLL(_SO_PATH)

# void max_dp_batch(scores, mask, N, dec, kp, B, L)
_lib.max_dp_batch.restype = None
_lib.max_dp_batch.argtypes = [
    ctypes.POINTER(ctypes.c_double),   # scores (B,L,L) float64
    ctypes.POINTER(ctypes.c_int8),     # mask   (B,L,L) int8
    ctypes.POINTER(ctypes.c_double),   # N      (B,L,L) float64 out
    ctypes.POINTER(ctypes.c_int8),     # dec    (B,L,L) int8 out
    ctypes.POINTER(ctypes.c_int64),    # kp     (B,L,L) int64 out
    ctypes.c_int,                      # B
    ctypes.c_int,                      # L
]

# void traceback_batch(dec, kp, B, L, pairs_out, counts_out)
_lib.traceback_batch.restype = None
_lib.traceback_batch.argtypes = [
    ctypes.POINTER(ctypes.c_int8),     # dec
    ctypes.POINTER(ctypes.c_int64),    # kp
    ctypes.c_int,                      # B
    ctypes.c_int,                      # L
    ctypes.POINTER(ctypes.c_int32),    # pairs_out (B, 2*L)
    ctypes.POINTER(ctypes.c_int32),    # counts_out (B,)
]

MIN_LOOP = 3


def nussinov_map_batch(scores_batch: np.ndarray, mask_batch: np.ndarray):
    """Batched decode for B sequences of the same length L.

    Parameters:
        scores_batch: (B, L, L) float64, -inf on illegal pairs
        mask_batch:   (B, L, L) bool

    Returns:
        list of B lists of (i, j) pairs, sorted
    """
    scores_batch = np.ascontiguousarray(scores_batch, dtype=np.float64)
    mask_batch = np.ascontiguousarray(mask_batch, dtype=np.int8)
    B, L, _ = scores_batch.shape

    N = np.zeros((B, L, L), dtype=np.float64)
    dec = np.zeros((B, L, L), dtype=np.int8)
    kp = np.full((B, L, L), -1, dtype=np.int64)

    _lib.max_dp_batch(
        scores_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        mask_batch.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        N.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        dec.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        kp.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        B, L
    )

    # Batched traceback
    pairs_flat = np.zeros((B, 2 * L), dtype=np.int32)
    counts = np.zeros(B, dtype=np.int32)

    _lib.traceback_batch(
        dec.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        kp.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        B, L,
        pairs_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
    )

    result = []
    for b in range(B):
        n = int(counts[b])
        pairs = [(int(pairs_flat[b, 2*k]), int(pairs_flat[b, 2*k+1]))
                 for k in range(n)]
        pairs.sort()
        result.append(pairs)
    return result


def nussinov_map_single(scores: np.ndarray, mask: np.ndarray):
    """Single-sequence decode via C extension (for verification/drop-in)."""
    scores = np.ascontiguousarray(scores[np.newaxis], dtype=np.float64)
    mask_b = np.ascontiguousarray(mask[np.newaxis], dtype=np.int8)
    return nussinov_map_batch(scores, mask_b)[0]


def test_bitwise_equal(L=50, seed=42, n_tests=20):
    """Verify C output matches Python reference bit-for-bit."""
    sys.path.insert(0, '/home/cunyuliu/rna-jepa/src')
    from rnajepa.harness import _max_dp_reference, _traceback_max, MIN_LOOP

    rng = np.random.default_rng(seed)
    all_ok = True
    for t in range(n_tests):
        # random scores with -inf on illegal pairs
        sc = rng.normal(0, 1, (L, L)).astype(np.float64)
        mk = rng.random((L, L)) > 0.3  # ~70% legal
        # apply min_loop constraint
        for i in range(L):
            for j in range(L):
                if j - i <= MIN_LOOP:
                    mk[i, j] = False
                    sc[i, j] = -np.inf
                if not mk[i, j]:
                    sc[i, j] = -np.inf

        # Python reference
        N_ref, dec_ref, kp_ref = _max_dp_reference(sc, mk)
        pairs_ref = _traceback_max(dec_ref, kp_ref, L)
        pairs_ref = sorted(pairs_ref)

        # C extension
        pairs_c = nussinov_map_single(sc, mk)

        if pairs_ref != pairs_c:
            print(f'  MISMATCH test {t}: py={len(pairs_ref)} pairs, c={len(pairs_c)} pairs')
            # find first diff
            ref_set = set(pairs_ref)
            c_set = set(pairs_c)
            only_ref = ref_set - c_set
            only_c = c_set - ref_set
            if only_ref:
                print(f'    only in py: {list(only_ref)[:5]}')
            if only_c:
                print(f'    only in c:  {list(only_c)[:5]}')
            all_ok = False
        else:
            # Also check N table bitwise
            scores_b = sc[np.newaxis]
            mask_b = mk.astype(np.int8)[np.newaxis]
            N_c = np.zeros((1, L, L), dtype=np.float64)
            dec_c = np.zeros((1, L, L), dtype=np.int8)
            kp_c = np.full((1, L, L), -1, dtype=np.int64)
            _lib.max_dp_batch(
                scores_b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                mask_b.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                N_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                dec_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                kp_c.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
                1, L
            )
            if not np.array_equal(N_ref, N_c[0]):
                diff = np.argwhere(N_ref != N_c[0])
                print(f'  N MISMATCH test {t}: {len(diff)} diffs, first: {diff[:3]}')
                all_ok = False
            if not np.array_equal(dec_ref, dec_c[0]):
                diff = np.argwhere(dec_ref != dec_c[0])
                print(f'  dec MISMATCH test {t}: {len(diff)} diffs')
                all_ok = False
            if not np.array_equal(kp_ref, kp_c[0]):
                diff = np.argwhere(kp_ref != kp_c[0])
                print(f'  kp MISMATCH test {t}: {len(diff)} diffs')
                all_ok = False

    if all_ok:
        print(f'ALL {n_tests} TESTS PASS: bitwise-equal (pairs + N + dec + kp)')
    else:
        print(f'SOME TESTS FAILED')
    return all_ok


if __name__ == '__main__':
    import time
    print('=== Bitwise equality test ===')
    test_bitwise_equal(L=50, seed=42, n_tests=20)
    test_bitwise_equal(L=100, seed=123, n_tests=10)
    test_bitwise_equal(L=200, seed=456, n_tests=5)

    print()
    print('=== Speed benchmark (single seq, L=200) ===')
    rng = np.random.default_rng(99)
    L = 200
    sc = rng.normal(0, 1, (L, L)).astype(np.float64)
    mk = np.ones((L, L), dtype=bool)
    for i in range(L):
        for j in range(L):
            if j - i <= MIN_LOOP:
                mk[i, j] = False
                sc[i, j] = -np.inf
            if not mk[i, j]:
                sc[i, j] = -np.inf

    sys.path.insert(0, '/home/cunyuliu/rna-jepa/src')
    from rnajepa.harness import nussinov_map as py_decode

    # warm up
    py_decode(sc, mk)
    nussinov_map_single(sc, mk)

    # Python
    t0 = time.perf_counter()
    for _ in range(10):
        py_decode(sc, mk)
    t_py = (time.perf_counter() - t0) / 10
    print(f'  Python (numpy): {t_py*1000:.1f} ms/seq')

    # C
    t0 = time.perf_counter()
    for _ in range(10):
        nussinov_map_single(sc, mk)
    t_c = (time.perf_counter() - t0) / 10
    print(f'  C (ctypes):     {t_c*1000:.1f} ms/seq')
    print(f'  Speedup: {t_py/t_c:.1f}x')

    # Batched
    B = 32
    sc_batch = np.stack([sc] * B)
    mk_batch = np.stack([mk] * B)
    t0 = time.perf_counter()
    nussinov_map_batch(sc_batch, mk_batch)
    t_batch = (time.perf_counter() - t0)
    print(f'  C batched B={B}: {t_batch*1000:.1f} ms total, {t_batch/B*1000:.1f} ms/seq')
    print(f'  Batched speedup vs Python: {t_py / (t_batch/B):.1f}x')
