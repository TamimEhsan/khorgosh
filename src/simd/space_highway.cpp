// Portable (Highway) raw float32 distance kernels. Compiled once; Highway's
// foreach_target.h re-includes this file per target internally, and
// HWY_DYNAMIC_DISPATCH below picks the best one at runtime (SSE4/AVX3 on
// x86, NEON/SVE on ARM, WASM SIMD, RVV, or HWY_SCALAR). See
// docs/portability/highway-plan.md Phase 2.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/space_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

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

// Sums query[i] for every dimension i whose bit is set in the packed code
// `data`, matching mask_ip_x0_q_avx2's bit convention exactly: dimension i's
// bit lives at position (63 - i % 64) of the 64-bit word at data[i/64] (MSB
// of each word first), i.e. word = memcpy'd 8 bytes starting at
// data + (i/64)*8, tested via (word >> (63 - i%64)) & 1. Verified against
// MaskIpX0Q.Avx2PreservesEveryStoredBitPosition's bit-for-bit convention.
// The per-lane bit test is scalar (this is not the hottest path relative to
// the O(padded_dim) float accumulation it feeds); only the summation is
// vectorized.
float MaskIpX0QImpl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT data, size_t padded_dim
) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    auto sum = hn::Zero(d);

    auto bit_at = [&](size_t pos) -> bool {
        uint64_t word = 0;
        std::memcpy(&word, data + (pos / 64) * 8, sizeof(word));
        return ((word >> (63 - pos % 64)) & 1U) != 0;
    };

    size_t i = 0;
    for (; i + lanes <= padded_dim; i += lanes) {
        float selector[hn::MaxLanes(d)];
        for (size_t lane = 0; lane < lanes; ++lane) {
            selector[lane] = bit_at(i + lane) ? 1.0F : 0.0F;
        }
        sum = hn::MulAdd(hn::LoadU(d, query + i), hn::LoadU(d, selector), sum);
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

// Transposes per-dimension b_query-bit codes into b_query bit-plane arrays:
// bit_idx's plane, word i/64, bit (63 - i%64) = bit bit_idx of q[i]. This
// output is purely transient (never persisted, never compared across
// backends — no existing test exercises new_transpose_bin[_512] directly),
// consumed only by warmup_ip_x0_q_512_highway below, so the two only need to
// agree with each other; they share this convention deliberately, matching
// mask_ip_x0_q_highway's bit-position convention above for data[] read by
// other kernels (word i/64, bit 63 - i%64). One-time per-query setup cost,
// not the search hot path, so this is plain scalar.
template <typename QCode>
void TransposeBinGeneric(
    const QCode* HWY_RESTRICT q,
    uint64_t* HWY_RESTRICT tq,
    size_t padded_dim,
    size_t b_query
) {
    const size_t num_words = padded_dim / 64;
    std::memset(tq, 0, sizeof(uint64_t) * num_words * b_query);
    for (size_t i = 0; i < padded_dim; ++i) {
        const QCode code = q[i];
        const size_t word = i / 64;
        const uint64_t bit_mask = uint64_t{1} << (63 - i % 64);
        for (size_t bit_idx = 0; bit_idx < b_query; ++bit_idx) {
            if ((code >> bit_idx) & 1U) {
                tq[bit_idx * num_words + word] |= bit_mask;
            }
        }
    }
}

void new_transpose_bin_highway(
    const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    TransposeBinGeneric(q, tq, padded_dim, b_query);
}

void new_transpose_bin_512_highway(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    TransposeBinGeneric(q, tq, padded_dim, b_query);
}

}  // namespace rabitqlib::simd
#endif  // HWY_ONCE
