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

}  // namespace rabitqlib::simd
#endif  // HWY_ONCE
