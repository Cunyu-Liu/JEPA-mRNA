/* dp_decode_c.c — C implementation of max-product Nussinov DP + traceback.
 *
 * Exact bitwise match to _max_dp_reference in harness.py:
 *   Recurrence:
 *     N(i,j) = max( N(i,j-1),                      // dec=0 unpaired-j
 *                   N(i+1,j),                      // dec=1 unpaired-i
 *                   N(i+1,j-1)+s_ij if mask[i,j],   // dec=2 pair(i,j)
 *                   max_k N(i,k-1)+N(k+1,j-1)+s_kj ) // dec=3 pair(k,j)
 *   Tie-breaking: first strictly-greater wins (later candidate only replaces
 *   if strictly > current best; argmax over k gives smallest k on ties).
 *
 * OpenMP-parallelised across batch dimension B.
 * Compile: gcc -O3 -fopenmp -shared -fPIC -o dp_decode_c.so dp_decode_c.c -lm
 */
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#define MIN_LOOP 3

/* Batched max-DP: process B independent sequences of the same length L.
 *
 * scores: (B, L, L) float64, row-major
 * mask:   (B, L, L) int8 (0/1), row-major
 * N:      (B, L, L) float64, output (caller allocates, zeroed)
 * dec:    (B, L, L) int8, output
 * kp:     (B, L, L) int64, output (init to -1)
 */
void max_dp_batch(
    const double *scores, const int8_t *mask,
    double *N, int8_t *dec, int64_t *kp,
    int B, int L)
{
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; b++) {
        const double *sc = scores + (size_t)b * L * L;
        const int8_t  *mk = mask   + (size_t)b * L * L;
        double        *Nb = N      + (size_t)b * L * L;
        int8_t        *db = dec    + (size_t)b * L * L;
        int64_t       *kb = kp     + (size_t)b * L * L;

        for (int d = 1; d < L; d++) {
            if (d <= MIN_LOOP) continue;
            int n_k = d - MIN_LOOP - 1;
            for (int i = 0; i < L - d; i++) {
                int j = i + d;
                size_t idx = (size_t)i * L + j;

                /* dec=0: unpaired-j */
                double best = Nb[(size_t)i * L + (j-1)];
                int8_t bdec = 0;
                int64_t bk = -1;

                /* dec=1: unpaired-i (strict >) */
                double v = Nb[(size_t)(i+1) * L + j];
                if (v > best) { best = v; bdec = 1; bk = -1; }

                /* dec=2: pair(i,j) if mask[i,j] (strict >) */
                if (mk[idx]) {
                    v = Nb[(size_t)(i+1) * L + (j-1)] + sc[idx];
                    if (v > best) { best = v; bdec = 2; bk = -1; }
                }

                /* dec=3: partner search k in [i+1, j-MIN_LOOP-1] */
                if (n_k > 0) {
                    double best_k = -INFINITY;
                    int best_ki = -1;
                    for (int k = i + 1; k <= j - MIN_LOOP - 1; k++) {
                        size_t kidx = (size_t)k * L + j;
                        if (!mk[kidx]) continue;
                        /* N(i,k-1) + N(k+1,j-1) + s_kj */
                        double vk;
                        /* handle k-1 < i (N=0) and k+1 > j-1 (N=0) */
                        double left, right;
                        left  = (k - 1 >= i) ? Nb[(size_t)i * L + (k-1)] : 0.0;
                        right = (k + 1 <= j - 1) ? Nb[(size_t)(k+1) * L + (j-1)] : 0.0;
                        vk = left + right + sc[kidx];
                        if (isnan(vk)) continue;
                        /* first strictly greater → keep smallest k on ties */
                        if (vk > best_k) { best_k = vk; best_ki = k; }
                    }
                    if (best_ki >= 0 && best_k > best) {
                        best = best_k; bdec = 3; bk = best_ki;
                    }
                }

                Nb[idx] = best;
                db[idx] = bdec;
                kb[idx] = bk;
            }
        }
    }
}

/* Traceback: iterative, exact match to _traceback_max in harness.py.
 * Returns pairs as flattened (i,j) array of int32.
 * Caller provides pairs_out (2*L int32) and returns count.
 */
int traceback_one(
    const int8_t *dec, const int64_t *kp,
    int L, int32_t *pairs_out)
{
    /* iterative stack-based traceback */
    int n_pairs = 0;
    /* stack of (i,j) intervals; max depth = L */
    int *stack_i = (int *)malloc(L * sizeof(int));
    int *stack_j = (int *)malloc(L * sizeof(int));
    int sp = 0;
    stack_i[sp] = 0; stack_j[sp] = L - 1; sp++;

    while (sp > 0) {
        sp--;
        int i = stack_i[sp];
        int j = stack_j[sp];
        if (i >= j || j - i <= MIN_LOOP) continue;
        size_t idx = (size_t)i * L + j;
        int8_t d = dec[idx];
        if (d == 0) {
            stack_i[sp] = i; stack_j[sp] = j - 1; sp++;
        } else if (d == 1) {
            stack_i[sp] = i + 1; stack_j[sp] = j; sp++;
        } else if (d == 2) {
            pairs_out[2 * n_pairs] = i;
            pairs_out[2 * n_pairs + 1] = j;
            n_pairs++;
            stack_i[sp] = i + 1; stack_j[sp] = j - 1; sp++;
        } else { /* d == 3 */
            int64_t k = kp[idx];
            pairs_out[2 * n_pairs] = (int32_t)k;
            pairs_out[2 * n_pairs + 1] = j;
            n_pairs++;
            stack_i[sp] = i; stack_j[sp] = (int)k - 1; sp++;
            stack_i[sp] = (int)k + 1; stack_j[sp] = j - 1; sp++;
        }
    }
    free(stack_i);
    free(stack_j);
    return n_pairs;
}

/* Batched traceback wrapper: process B sequences.
 * pairs_out: (B, 2*max_pairs) int32, caller allocates
 * counts_out: (B,) int32, number of pairs per sequence
 */
void traceback_batch(
    const int8_t *dec, const int64_t *kp,
    int B, int L,
    int32_t *pairs_out, int32_t *counts_out)
{
    #pragma omp parallel for schedule(dynamic)
    for (int b = 0; b < B; b++) {
        const int8_t *db = dec + (size_t)b * L * L;
        const int64_t *kb = kp  + (size_t)b * L * L;
        int32_t *pb = pairs_out + (size_t)b * 2 * L;
        counts_out[b] = traceback_one(db, kb, L, pb);
    }
}
