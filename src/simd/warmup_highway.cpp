// Portable (Highway) warmup/cold-start estimation kernel. See
// docs/portability/highway-plan.md Phase 3.
//
// This is plain scalar, word-at-a-time popcount, not a Highway vector
// kernel: modern hardware executes POPCNT on a 64-bit general-purpose
// register in ~1-3 cycles (rabitqlib::bitops::popcount64 already wraps the
// portable __builtin_popcountll/MSVC intrinsic), so there is no vector-width
// dependent code here for Highway's foreach_target.h/HWY_DYNAMIC_DISPATCH
// machinery to select between. The AVX2/AVX-512 kernels' vectorized
// popcount lookup-table tricks exist only to work around AVX2 lacking a
// vector POPCNT instruction (added only in AVX-512 VPOPCNTDQ); a portable
// per-word scalar loop sidesteps that problem entirely.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/bitops.hpp"

namespace rabitqlib::simd {

// data: padded_dim/8 bytes, one bit per dimension (bin_code's existing,
// shared packed-code format; see mask_ip_x0_q_highway's file comment for the
// bit convention word/bit indexing shared across kernels reading bin_code).
// query: b_query bit-planes of padded_dim/64 words each, laid out as
// query[bit_idx * (padded_dim/64) + word] — exactly new_transpose_bin_512_
// highway's output layout (see space_highway.cpp); this pairing is
// transient/per-query and never compared cross-backend, so the two need
// only agree with each other, which they do by construction.
//
// ip = sum_i data_bit[i] * query_value[i], where query_value[i] is the
// b_query-bit integer reconstructed from its bit-planes, computed as
// sum_bit_idx (popcount(data_word & plane_word) << bit_idx) accumulated
// per word — the standard bit-sliced dot product technique. ppc is the
// total population count of data, used for the delta/vl dequantization
// below (matching warmup_ip_x0_q_512_avx2 exactly).
float warmup_ip_x0_q_512_highway(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    const size_t num_words = padded_dim / 64;
    uint64_t ppc = 0;
    uint64_t ip = 0;
    for (size_t word = 0; word < num_words; ++word) {
        uint64_t data_word = 0;
        std::memcpy(&data_word, data + word * 8, sizeof(data_word));
        ppc += bitops::popcount64(data_word);
        for (size_t bit_idx = 0; bit_idx < b_query; ++bit_idx) {
            const uint64_t plane_word = query[bit_idx * num_words + word];
            ip += static_cast<uint64_t>(bitops::popcount64(data_word & plane_word))
                  << bit_idx;
        }
    }
    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}

float warmup_ip_x0_q_512_highway(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return warmup_ip_x0_q_512_highway(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, padded_dim, b_query
    );
}

}  // namespace rabitqlib::simd
