// Portable (Highway) scalar quantization kernels. See
// docs/portability/highway-plan.md Phase 3.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/quantization_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "rabitqlib/simd/space_dispatch.hpp"

HWY_BEFORE_NAMESPACE();
namespace rabitqlib::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// result[i] = round_half_away_from_zero(clamp((vec0[i] - lo) / delta, 0,
// max<T>)), matching scalar_quantize_uint{8,16}_avx2's saturating-pack
// behavior. hn::ConvertTo truncates rather than rounding (verified), so
// rounding happens explicitly in the float domain first: Trunc + compare the
// fractional part against 0.5. Values are clamped to be non-negative before
// this step, so "away from zero" and "half up" coincide and no sign handling
// is needed (unlike FlipSignImpl's use case).
template <typename T>
HWY_ATTR void QuantizeImpl(
    T* HWY_RESTRICT result,
    const float* HWY_RESTRICT vec0,
    size_t dim,
    float lo,
    float delta
) {
    const hn::ScalableTag<float> df;
    const hn::RebindToSigned<decltype(df)> di32;
    const size_t lanes = hn::Lanes(df);
    const float one_over_delta = 1.0F / delta;
    const float max_value = static_cast<float>(std::numeric_limits<T>::max());

    const auto lo_v = hn::Set(df, lo);
    const auto od_v = hn::Set(df, one_over_delta);
    const auto zero_v = hn::Zero(df);
    const auto max_v = hn::Set(df, max_value);
    const auto half_v = hn::Set(df, 0.5F);
    const auto one_v = hn::Set(df, 1.0F);

    size_t i = 0;
    for (; i + lanes <= dim; i += lanes) {
        auto x = hn::Mul(hn::Sub(hn::LoadU(df, vec0 + i), lo_v), od_v);
        x = hn::Max(hn::Min(x, max_v), zero_v);
        const auto truncated = hn::Trunc(x);
        const auto frac = hn::Sub(x, truncated);
        const auto round_up = hn::Ge(frac, half_v);
        const auto rounded = hn::Add(truncated, hn::IfThenElseZero(round_up, one_v));
        const auto as_i32 = hn::ConvertTo(di32, rounded);

        if constexpr (sizeof(T) == 1) {
            const hn::Rebind<uint16_t, decltype(di32)> du16;
            const hn::Rebind<uint8_t, decltype(du16)> du8;
            hn::StoreU(hn::DemoteTo(du8, hn::DemoteTo(du16, as_i32)), du8, result + i);
        } else {
            static_assert(sizeof(T) == 2);
            const hn::Rebind<uint16_t, decltype(di32)> du16;
            hn::StoreU(hn::DemoteTo(du16, as_i32), du16, result + i);
        }
    }
    for (; i < dim; ++i) {
        float x = (vec0[i] - lo) * one_over_delta;
        x = x < 0.0F ? 0.0F : (x > max_value ? max_value : x);
        result[i] = static_cast<T>(std::round(x));
    }
}

void ScalarQuantizeUint8Impl(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    QuantizeImpl<uint8_t>(result, vec0, dim, lo, delta);
}

void ScalarQuantizeUint16Impl(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    QuantizeImpl<uint16_t>(result, vec0, dim, lo, delta);
}

}  // namespace HWY_NAMESPACE
}  // namespace rabitqlib::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rabitqlib::simd {

HWY_EXPORT(ScalarQuantizeUint8Impl);
HWY_EXPORT(ScalarQuantizeUint16Impl);

void scalar_quantize_uint8_highway(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    HWY_DYNAMIC_DISPATCH(ScalarQuantizeUint8Impl)(result, vec0, dim, lo, delta);
}

void scalar_quantize_uint16_highway(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    HWY_DYNAMIC_DISPATCH(ScalarQuantizeUint16Impl)(result, vec0, dim, lo, delta);
}

}  // namespace rabitqlib::simd
#endif  // HWY_ONCE
