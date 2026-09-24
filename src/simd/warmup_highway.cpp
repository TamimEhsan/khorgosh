// Portable (Highway) warmup/cold-start estimation kernel. See
// docs/portability/highway-plan.md Phase 3.
//
// warmup_ip_x0_q_512 is called once per visited HNSW candidate (the cheap
// first-pass distance gate — see estimator.hpp's split_single_estdist_direct),
// so it is a genuine search hot path, not a one-time setup cost. An earlier
// version of this file was plain scalar, word-at-a-time popcount, reasoning
// that hardware POPCNT needs no vector-width dispatch — true for the
// instruction itself, but that reasoning missed that staying outside
// Highway's foreach_target/HWY_DYNAMIC_DISPATCH machinery also means never
// getting per-target compile flags: profiling a portable build
// (RABITQ_ENABLE_NATIVE_OPTIMIZATION=OFF, matching AGENTS.md's requirement
// for portable binaries/wheels) showed ~33% of total query time inside
// __popcountdi2, GCC's software popcount fallback, because the file had no
// -march flag at all and __builtin_popcountll therefore couldn't assume
// hardware POPCNT was available. hn::PopulationCount fixes this properly:
// on x86_256/x86_512 targets below AVX3_DL (i.e. without hardware
// VPOPCNTDQ) it already uses the same vectorized nibble-shuffle-table
// technique as warmup_avx2.cpp's hand-written popcount_avx2 (see
// x86_256-inl.h/x86_512-inl.h), and native VPOPCNTDQ once available — so
// this now gets genuine SIMD popcount on every target, not just "the
// hardware instruction when the compiler happens to allow it".
//
// query layout matches new_transpose_bin_512_highway's documented
// block/chunk convention exactly (see that function's comment in
// space_highway.cpp) — this is warmup_ip_x0_q_512's public, cross-backend
// contract (verified by WarmupIpX0Q.SupportsUnalignedCodes in
// space_test.cpp), not a free internal choice.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/warmup_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "rabitqlib/simd/warmup_dispatch.hpp"

HWY_BEFORE_NAMESPACE();
namespace rabitqlib::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Mirrors warmup_ip_x0_q_512_avx2's structure exactly: blocks of up to 8
// 64-bit words (512 bits), one popcount accumulator per query bit-plane
// (acc_bits[bit_idx]), shifted by bit_idx and summed only after the whole
// buffer is processed (the shift is per bit-plane total, not per word).
// b_query > kMaxQueryBits is rejected by the caller below, matching
// warmup_ip_x0_q_512_avx2's own bound. Declared per-target (this namespace
// is re-entered once per Highway target) rather than at file scope, since
// the latter would redefine it on every foreach_target.h re-inclusion
// within this one translation unit; the HWY_ONCE wrapper below uses its own
// copy instead of reaching into a specific target's namespace for it.
constexpr size_t kMaxQueryBits = 8;
constexpr size_t kChunksPerBlock = 8;

HWY_ATTR float WarmupIpX0Q512Impl(
    const uint8_t* HWY_RESTRICT data,
    const uint64_t* HWY_RESTRICT query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    const hn::ScalableTag<uint64_t> d;
    const size_t lanes = hn::Lanes(d);
    const size_t num_words = padded_dim / 64;

    auto acc_ppc = hn::Zero(d);
    hn::Vec<decltype(d)> acc_bits[kMaxQueryBits];
    for (size_t bit_idx = 0; bit_idx < b_query; ++bit_idx) {
        acc_bits[bit_idx] = hn::Zero(d);
    }

    size_t word = 0;
    size_t block_query_offset = 0;
    while (word < num_words) {
        const size_t chunks = std::min(kChunksPerBlock, num_words - word);
        for (size_t c = 0; c < chunks; c += lanes) {
            const size_t cnt = std::min(lanes, chunks - c);
            const auto* data_ptr =
                reinterpret_cast<const uint64_t*>(data + ((word + c) * 8));
            const auto dv =
                (cnt == lanes) ? hn::LoadU(d, data_ptr) : hn::LoadN(d, data_ptr, cnt);
            acc_ppc = hn::Add(acc_ppc, hn::PopulationCount(dv));
            for (size_t bit_idx = 0; bit_idx < b_query; ++bit_idx) {
                const uint64_t* qptr = query + block_query_offset + (bit_idx * chunks) + c;
                const auto qv =
                    (cnt == lanes) ? hn::LoadU(d, qptr) : hn::LoadN(d, qptr, cnt);
                acc_bits[bit_idx] =
                    hn::Add(acc_bits[bit_idx], hn::PopulationCount(hn::And(dv, qv)));
            }
        }
        word += chunks;
        block_query_offset += chunks * b_query;
    }

    auto acc_ip = hn::Zero(d);
    for (size_t bit_idx = 0; bit_idx < b_query; ++bit_idx) {
        acc_ip = hn::Add(
            acc_ip, hn::ShiftLeftSame(acc_bits[bit_idx], static_cast<int>(bit_idx))
        );
    }

    const auto ip = static_cast<float>(hn::ReduceSum(d, acc_ip));
    const auto ppc = static_cast<float>(hn::ReduceSum(d, acc_ppc));
    return (delta * ip) + (vl * ppc);
}

}  // namespace HWY_NAMESPACE
}  // namespace rabitqlib::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rabitqlib::simd {

HWY_EXPORT(WarmupIpX0Q512Impl);

float warmup_ip_x0_q_512_highway(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    // Matches HWY_NAMESPACE::kMaxQueryBits above (and
    // warmup_ip_x0_q_512_avx2's own bound); not reachable from here since
    // that namespace is target-specific.
    constexpr size_t kMaxQueryBits = 8;
    if (b_query > kMaxQueryBits) {
        throw std::invalid_argument("warmup_ip_x0_q_512 requires at most 8 query bits");
    }
    return HWY_DYNAMIC_DISPATCH(WarmupIpX0Q512Impl
    )(data, query, delta, vl, padded_dim, b_query);
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
#endif  // HWY_ONCE
