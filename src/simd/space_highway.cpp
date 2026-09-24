// Portable (Highway) raw float32 distance kernels. Compiled once; Highway's
// foreach_target.h re-includes this file per target internally, and
// HWY_DYNAMIC_DISPATCH below picks the best one at runtime (SSE4/AVX3 on
// x86, NEON/SVE on ARM, WASM SIMD, RVV, or HWY_SCALAR). See
// docs/portability/highway-plan.md Phase 2.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/space_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rabitqlib/simd/space_dispatch.hpp"

HWY_BEFORE_NAMESPACE();
namespace rabitqlib::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

enum class FloatOp { kSquaredL2, kDot, kSquaredNorm };

// Four independent accumulators avoid a single FMA dependency chain, the
// same reasoning as the hand-written AVX2/AVX-512 kernels in
// space_float_kernels.hpp. hn::LoadN safely loads the final partial vector
// without reading past `n` elements (required: see
// FloatDistance.TailsDoNotReadPastGuardPage).
template <FloatOp kOp>
HWY_ATTR float RawFloat(
    const float* HWY_RESTRICT a, const float* HWY_RESTRICT b, size_t n
) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    auto s0 = hn::Zero(d);
    auto s1 = hn::Zero(d);
    auto s2 = hn::Zero(d);
    auto s3 = hn::Zero(d);

    auto accumulate = [&](decltype(s0) sum, size_t at, size_t count) {
        const auto x =
            (count == lanes) ? hn::LoadU(d, a + at) : hn::LoadN(d, a + at, count);
        if constexpr (kOp == FloatOp::kSquaredNorm) {
            return hn::MulAdd(x, x, sum);
        } else {
            const auto y =
                (count == lanes) ? hn::LoadU(d, b + at) : hn::LoadN(d, b + at, count);
            if constexpr (kOp == FloatOp::kSquaredL2) {
                const auto diff = hn::Sub(x, y);
                return hn::MulAdd(diff, diff, sum);
            } else {
                return hn::MulAdd(x, y, sum);
            }
        }
    };

    size_t i = 0;
    for (; i + lanes * 4 <= n; i += lanes * 4) {
        s0 = accumulate(s0, i, lanes);
        s1 = accumulate(s1, i + lanes, lanes);
        s2 = accumulate(s2, i + lanes * 2, lanes);
        s3 = accumulate(s3, i + lanes * 3, lanes);
    }
    auto sum = hn::Add(hn::Add(s0, s1), hn::Add(s2, s3));
    for (; i + lanes <= n; i += lanes) {
        sum = accumulate(sum, i, lanes);
    }
    if (i < n) {
        sum = accumulate(sum, i, n - i);
    }
    return hn::ReduceSum(d, sum);
}

float EuclideanSqrImpl(const float* a, const float* b, size_t n) {
    return RawFloat<FloatOp::kSquaredL2>(a, b, n);
}

float DotProductImpl(const float* a, const float* b, size_t n) {
    return RawFloat<FloatOp::kDot>(a, b, n);
}

float L2NormSqrImpl(const float* a, size_t n) {
    return RawFloat<FloatOp::kSquaredNorm>(a, a, n);
}

// Standard SWAR bit reversal (matches rabitqlib::reverse_bits_u64 in
// utils/space.hpp, duplicated locally rather than pulling in that header's
// unrelated Eigen-based machinery for one six-line helper).
inline uint64_t ReverseBits64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555ULL) | ((n << 1) & 0xaaaaaaaaaaaaaaaaULL);
    n = ((n >> 2) & 0x3333333333333333ULL) | ((n << 2) & 0xccccccccccccccccULL);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0fULL) | ((n << 4) & 0xf0f0f0f0f0f0f0f0ULL);
    n = ((n >> 8) & 0x00ff00ff00ff00ffULL) | ((n << 8) & 0xff00ff00ff00ff00ULL);
    n = ((n >> 16) & 0x0000ffff0000ffffULL) | ((n << 16) & 0xffff0000ffff0000ULL);
    n = ((n >> 32) & 0x00000000ffffffffULL) | ((n << 32) & 0xffffffff00000000ULL);
    return n;
}

// Sums query[i] for every dimension i whose bit is set in the packed code
// `data`, matching mask_ip_x0_q_avx2's bit convention exactly: dimension i's
// bit lives at position (63 - i % 64) of the 64-bit word at data[i/64] (MSB
// of each word first), i.e. word = memcpy'd 8 bytes starting at
// data + (i/64)*8, tested via (word >> (63 - i%64)) & 1. Verified against
// MaskIpX0Q.Avx2PreservesEveryStoredBitPosition's bit-for-bit convention.
//
// This is a search hot path (called once per full-distance HNSW candidate).
// An earlier version vectorized bit extraction via broadcast+variable-shift
// (still correct, ~1.6x faster than the original scalar version), but
// mask_ip_x0_q_avx512 is faster still: it reverses each 64-bit word ONCE
// (turning the MSB-first storage into a plain LSB-first bitmask), then
// reinterprets 16-bit slices of that reversed word directly as __mmask16
// and uses _mm512_maskz_loadu_ps — a single native masked-load instruction,
// with no broadcast/shift/compare needed at all. This mirrors that
// technique with hn::LoadMaskBits (builds a mask directly from packed bits
// — the portable equivalent of casting an integer to a mask register) and
// hn::MaskedLoad (the portable equivalent of _mm512_maskz_loadu_ps), which
// map to the same native instructions on HWY_TARGET <= HWY_AVX3. Processes
// a full 64-bit word (padded_dim's rotation always pads to a multiple of
// 64) per outer iteration, split into padded_dim/64/lanes lanes-wide groups
// — same "64 % lanes == 0, else fall back to scalar" guard as before, for
// the same reason.
float MaskIpX0QImpl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT data, size_t padded_dim
) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);

    auto bit_at = [&](size_t pos) -> bool {
        uint64_t word = 0;
        std::memcpy(&word, data + (pos / 64) * 8, sizeof(word));
        return ((word >> (63 - pos % 64)) & 1U) != 0;
    };

    auto sum = hn::Zero(d);
    size_t i = 0;

    if (lanes != 0 && lanes <= 64 && 64 % lanes == 0) {
        const size_t word_end = (padded_dim / 64) * 64;
        for (; i < word_end; i += 64) {
            uint64_t word = 0;
            std::memcpy(&word, data + (i / 64) * 8, sizeof(word));
            const uint64_t reversed = ReverseBits64(word);
            for (size_t g = 0; g < 64; g += lanes) {
                const uint64_t group_bits = reversed >> g;
                uint8_t bits_buf[8];
                std::memcpy(bits_buf, &group_bits, sizeof(bits_buf));
                const auto mask = hn::LoadMaskBits(d, bits_buf);
                sum = hn::Add(sum, hn::MaskedLoad(mask, d, query + i + g));
            }
        }
    }

    float result = hn::ReduceSum(d, sum);
    for (; i < padded_dim; ++i) {
        if (bit_at(i)) {
            result += query[i];
        }
    }
    return result;
}

}  // namespace HWY_NAMESPACE
}  // namespace rabitqlib::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rabitqlib::simd {

HWY_EXPORT(EuclideanSqrImpl);
HWY_EXPORT(DotProductImpl);
HWY_EXPORT(L2NormSqrImpl);

float euclidean_sqr_highway(const float* a, const float* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(EuclideanSqrImpl)(a, b, dim);
}

float dot_product_highway(const float* a, const float* b, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(DotProductImpl)(a, b, dim);
}

float dot_product_dis_highway(const float* a, const float* b, size_t dim) {
    return 1.0F - dot_product_highway(a, b, dim);
}

float l2norm_sqr_highway(const float* a, size_t dim) {
    return HWY_DYNAMIC_DISPATCH(L2NormSqrImpl)(a, dim);
}

HWY_EXPORT(MaskIpX0QImpl);

float mask_ip_x0_q_highway(const float* query, const uint8_t* data, size_t padded_dim) {
    return HWY_DYNAMIC_DISPATCH(MaskIpX0QImpl)(query, data, padded_dim);
}

float mask_ip_x0_q_highway(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q_highway(query, reinterpret_cast<const uint8_t*>(data), padded_dim);
}

// Transposes per-dimension b_query-bit codes into b_query bit-plane arrays,
// grouped into blocks of up to kChunksPerBlock 64-wide chunks: within block
// b (chunks c_0 .. c_n-1, global word index `word = block_first_word +
// chunk`), bit_idx's plane for that block starts at
// tq[block_tq_offset + bit_idx * n], and word `chunk`'s bit (63 - d) = bit
// bit_idx of q[word * 64 + d]. Blocks are concatenated block-major
// (block_tq_offset accumulates n * b_query per block).
//
// This is NOT a free internal choice: new_transpose_bin_512's output is
// read by the public warmup_ip_x0_q_512 entry point, which
// WarmupIpX0Q.SupportsUnalignedCodes (space_test.cpp) exercises against a
// query array hand-built to match new_transpose_bin_512_avx2's exact
// layout, independent of which backend the public entry point dispatches
// to. kChunksPerBlock=8 (block size 512) reproduces that AVX2 kernel's
// blocking (derived from new_transpose_bin_512_avx2's movemask/
// reverse_bits_u64 sequence: block_tq_offset advances by
// num_chunks_in_block * b_query, matching the test's `query_offset +=
// chunks * b_query`). kChunksPerBlock=1 reproduces new_transpose_bin_avx2's
// per-64-chunk layout (block size 64 degenerates every block to exactly one
// chunk, i.e. plain chunk-major with no larger blocking) — new_transpose_
// bin_highway has no caller elsewhere in the codebase and no dedicated
// cross-backend test, but this keeps it consistent with its own AVX2
// counterpart's documented behavior rather than an arbitrary convention.
// One-time per-query setup cost, not the search hot path, so this is plain
// scalar.
template <typename QCode, size_t kChunksPerBlock>
void TransposeBinBlocked(
    const QCode* HWY_RESTRICT q,
    uint64_t* HWY_RESTRICT tq,
    size_t padded_dim,
    size_t b_query
) {
    const size_t num_words = padded_dim / 64;
    std::memset(tq, 0, sizeof(uint64_t) * num_words * b_query);
    size_t word = 0;
    size_t block_tq_offset = 0;
    while (word < num_words) {
        const size_t chunks = std::min(kChunksPerBlock, num_words - word);
        for (size_t chunk = 0; chunk < chunks; ++chunk, ++word) {
            for (size_t d = 0; d < 64; ++d) {
                const QCode code = q[(word * 64) + d];
                const uint64_t bit_mask = uint64_t{1} << (63 - d);
                for (size_t bit_idx = 0; bit_idx < b_query; ++bit_idx) {
                    if ((code >> bit_idx) & 1U) {
                        tq[block_tq_offset + (bit_idx * chunks) + chunk] |= bit_mask;
                    }
                }
            }
        }
        block_tq_offset += chunks * b_query;
    }
}

void new_transpose_bin_highway(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    TransposeBinBlocked<uint16_t, 1>(q, tq, padded_dim, b_query);
}

void new_transpose_bin_512_highway(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    TransposeBinBlocked<uint8_t, 8>(q, tq, padded_dim, b_query);
}

}  // namespace rabitqlib::simd
#endif  // HWY_ONCE
