#pragma once

#include <algorithm>
#include <cstddef>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/space.hpp"

// One query against many rows: the shape that shows up wherever a vector is
// scored against a whole set of centroids.
//
// Done a row at a time, the query is re-read from memory for every row. Done as
// a matrix-vector product it stays in registers across rows, so the cost drops
// towards streaming the matrix once.
//
// The L2 helpers below expand
//
//     ||m - q||^2 = ||m||^2 - 2 <m, q> + ||q||^2
//
// which turns a distance computation into a dot product plus a lookup, at the
// price of precision: the right-hand side subtracts two large quantities to
// reach a small one, so it loses accuracy when the distance is small relative to
// the norms.
//
// Measured on GIST-960 -- query against 64 centroids and against 2000 base
// vectors -- the mean relative error is 1.2e-07 and the worst 6.3e-07, i.e. at
// float32 epsilon. The cancellation stays harmless there because the distances
// sit within a factor of ~4 of the squared norms. It would bite on
// near-duplicates, where the distance is orders of magnitude below the norm;
// gemv_test covers that case explicitly. If the result feeds a sort whose order
// matters (choosing nprobe clusters, say), verify recall rather than assuming.

namespace rabitqlib {

/**
 * @brief dots[i] = <mat_i, vec> for each of n rows.
 *
 * @param mat  n x dim, row major
 * @param vec  length dim
 * @param dots output, length n
 */
inline void gemv(const float* mat, const float* vec, size_t n, size_t dim, float* dots) {
    ConstRowMajorMatrixMap<float> m(mat, static_cast<long>(n), static_cast<long>(dim));
    ConstVectorMap<float> v(vec, static_cast<long>(dim));
    VectorMap<float> out(dots, static_cast<long>(n));
    out.noalias() = m * v;
}

/**
 * @brief norms_sqr[i] = ||mat_i||^2. Compute once when the matrix is set, not
 *        per query -- that is what makes the L2 expansion worth doing.
 */
inline void row_norms_sqr(const float* mat, size_t n, size_t dim, float* norms_sqr) {
    for (size_t i = 0; i < n; ++i) {
        ConstVectorMap<float> row(mat + (i * dim), static_cast<long>(dim));
        norms_sqr[i] = row.squaredNorm();
    }
}

/**
 * @brief dists[i] = ||mat_i - vec||^2 from precomputed dots and row norms.
 *
 * Clamped at zero: the expansion can land slightly negative when the true
 * distance is near zero and the cancellation goes the wrong way.
 */
inline void l2_from_dots(
    const float* dots, const float* norms_sqr, float vec_norm_sqr, size_t n, float* dists
) {
    for (size_t i = 0; i < n; ++i) {
        float d = norms_sqr[i] - (2.0F * dots[i]) + vec_norm_sqr;
        dists[i] = std::max(d, 0.0F);
    }
}

// Above this many matrix bytes the expansion stops paying, so l2_to_rows falls
// back to the direct per-row distance. See the note on l2_to_rows.
inline constexpr size_t kGemvL2ByteBudget = 2UL << 20;

/**
 * @brief dists[i] = ||mat_i - vec||^2 for each of n rows.
 *
 * Picks between the gemv expansion and a direct per-row euclidean_sqr by matrix
 * size. The expansion wins only while the matrix stays in L2: both paths stream
 * the matrix once, and the expansion's advantage is that the query stays in
 * registers -- worth 1.4-1.6x at dim=960 in cache, but nothing once the matrix
 * comes from DRAM and bandwidth sets the pace. Measured on this machine (2 MiB
 * L2/core, dim=960): 1.43x at 1.9 MiB, 0.98x at 2.9 MiB, 0.95x at 7.5 MiB.
 *
 * The fallback is also the exact one, so the large-n case avoids the expansion's
 * cancellation as a side effect.
 *
 * Callers wanting the dot products for their own use should call gemv directly;
 * this entry point is for callers that only want distances.
 *
 * @param norms_sqr precomputed row norms from row_norms_sqr()
 * @param scratch   length n, may alias dists
 */
inline void l2_to_rows(
    const float* mat,
    const float* norms_sqr,
    const float* vec,
    size_t n,
    size_t dim,
    float* dists,
    float* scratch
) {
    if (n * dim * sizeof(float) > kGemvL2ByteBudget) {
        for (size_t i = 0; i < n; ++i) {
            dists[i] = euclidean_sqr(vec, mat + (i * dim), dim);
        }
        return;
    }
    gemv(mat, vec, n, dim, scratch);
    ConstVectorMap<float> v(vec, static_cast<long>(dim));
    l2_from_dots(scratch, norms_sqr, v.squaredNorm(), n, dists);
}

}  // namespace rabitqlib
