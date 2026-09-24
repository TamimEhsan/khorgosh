// Portable (Highway) excode inner-product kernels (the read side of
// pack_excode_highway.cpp's packed layout). See
// docs/portability/highway-plan.md Phase 3.
//
// Each ip*_fxu* function is the exact inverse of the matching
// packing_Nbit_excode_highway formula (see that file's comments for the
// derivation of each packed layout, including the shared 3rd/5th/7th
// "top bit" plane convention) — this file does not re-derive anything new,
// it only unpacks. Codes are unpacked into a local float buffer per block,
// then accumulated with Highway's ordinary float dot-product pattern (the
// same MulAdd-based approach as space_highway.cpp's RawFloat), rather than
// keeping codes as narrow integers and using Highway's integer widening
// ops: this is search-hot-path code (unlike packing, which runs once at
// construction), so this still vectorizes the actual multiply-accumulate,
// while keeping the bit-unpacking scalar and simple to verify against the
// packer above.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/space_excode_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cstddef>
#include <cstdint>

#include "rabitqlib/simd/space_dispatch.hpp"

HWY_BEFORE_NAMESPACE();
namespace rabitqlib::simd::excode_ipimpl {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Adds the dot product of query[0, n) and codes[0, n) into `sum`. n (8, 16,
// or 64 at call sites below) need not be a multiple of the hardware's lane
// count: e.g. wide SVE can exceed 16 lanes for float, so the tail must be
// guarded with LoadN exactly like space_highway.cpp's RawFloat (reading
// past `codes`, a fixed-size local buffer, or past query's own allocation
// otherwise).
template <class D>
HWY_ATTR hn::Vec<D> DotAccumulate(
    D d,
    const float* HWY_RESTRICT query,
    const float* HWY_RESTRICT codes,
    size_t n,
    hn::Vec<D> sum
) {
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        sum = hn::MulAdd(hn::LoadU(d, query + i), hn::LoadU(d, codes + i), sum);
    }
    if (i < n) {
        sum =
            hn::MulAdd(hn::LoadN(d, query + i, n - i), hn::LoadN(d, codes + i, n - i), sum);
    }
    return sum;
}

// Same convention as mask_ip_x0_q_highway (space_highway.cpp), but this is
// an unrelated data structure (excode, not bin_code) with its own,
// LSB-first convention: byte holds dim (base+j)'s bit at bit position j
// (see ip16_fxu1_avx2's bitmask = {1,2,4,...,128}).
float IP16Fxu1Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT compact_code, size_t dim
) {
    const hn::ScalableTag<float> d;
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < dim; i += 8) {
        const uint8_t byte = compact_code[i / 8];
        float selector[8];
        for (size_t j = 0; j < 8; ++j) {
            selector[j] = ((byte >> j) & 1U) ? 1.0F : 0.0F;
        }
        sum = DotAccumulate(d, query + i, selector, 8, sum);
    }
    return hn::ReduceSum(d, sum);
}

float IP64Fxu2Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT compact_code, size_t dim
) {
    const hn::ScalableTag<float> d;
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < dim; i += 64) {
        float codes[64];
        for (size_t k = 0; k < 16; ++k) {
            const uint8_t byte = compact_code[k];
            codes[k] = static_cast<float>(byte & 0x3U);
            codes[16 + k] = static_cast<float>((byte >> 2) & 0x3U);
            codes[32 + k] = static_cast<float>((byte >> 4) & 0x3U);
            codes[48 + k] = static_cast<float>((byte >> 6) & 0x3U);
        }
        sum = DotAccumulate(d, query + i, codes, 64, sum);
        compact_code += 16;
    }
    return hn::ReduceSum(d, sum);
}

float IP64Fxu3Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT compact_code, size_t dim
) {
    const hn::ScalableTag<float> d;
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < dim; i += 64) {
        const uint8_t* low2 = compact_code;
        const uint8_t* top_bit = compact_code + 16;
        float codes[64];
        for (size_t c = 0; c < 64; ++c) {
            const size_t k = c % 16;
            const size_t group = c / 16;
            const uint8_t low = (low2[k] >> (group * 2)) & 0x3U;
            const uint8_t top = (top_bit[c % 8] >> (c / 8)) & 1U;
            codes[c] = static_cast<float>(low | (top << 2));
        }
        sum = DotAccumulate(d, query + i, codes, 64, sum);
        compact_code += 24;
    }
    return hn::ReduceSum(d, sum);
}

float IP16Fxu4Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT compact_code, size_t dim
) {
    const hn::ScalableTag<float> d;
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < dim; i += 16) {
        float codes[16];
        for (size_t k = 0; k < 8; ++k) {
            const uint8_t byte = compact_code[k];
            codes[k] = static_cast<float>(byte & 0xFU);
            codes[8 + k] = static_cast<float>((byte >> 4) & 0xFU);
        }
        sum = DotAccumulate(d, query + i, codes, 16, sum);
        compact_code += 8;
    }
    return hn::ReduceSum(d, sum);
}

float IP64Fxu5Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT compact_code, size_t dim
) {
    const hn::ScalableTag<float> d;
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < dim; i += 64) {
        const uint8_t* half1 = compact_code;
        const uint8_t* half2 = compact_code + 16;
        const uint8_t* top_bit = compact_code + 32;
        float codes[64];
        for (size_t k = 0; k < 16; ++k) {
            codes[k] = static_cast<float>(half1[k] & 0xFU);
            codes[16 + k] = static_cast<float>((half1[k] >> 4) & 0xFU);
            codes[32 + k] = static_cast<float>(half2[k] & 0xFU);
            codes[48 + k] = static_cast<float>((half2[k] >> 4) & 0xFU);
        }
        for (size_t c = 0; c < 64; ++c) {
            const uint8_t top = (top_bit[c % 8] >> (c / 8)) & 1U;
            codes[c] += static_cast<float>(top << 4);
        }
        sum = DotAccumulate(d, query + i, codes, 64, sum);
        compact_code += 40;
    }
    return hn::ReduceSum(d, sum);
}

float IP64Fxu6Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT compact_code, size_t dim
) {
    const hn::ScalableTag<float> d;
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < dim; i += 64) {
        const uint8_t* g1 = compact_code;
        const uint8_t* g2 = compact_code + 16;
        const uint8_t* g3 = compact_code + 32;
        float codes[64];
        for (size_t k = 0; k < 16; ++k) {
            codes[k] = static_cast<float>(g1[k] & 0x3FU);
            codes[16 + k] = static_cast<float>(g2[k] & 0x3FU);
            codes[32 + k] = static_cast<float>(g3[k] & 0x3FU);
            const uint8_t high = static_cast<uint8_t>(
                ((g1[k] >> 6) & 0x3U) | (((g2[k] >> 6) & 0x3U) << 2) |
                (((g3[k] >> 6) & 0x3U) << 4)
            );
            codes[48 + k] = static_cast<float>(high);
        }
        sum = DotAccumulate(d, query + i, codes, 64, sum);
        compact_code += 48;
    }
    return hn::ReduceSum(d, sum);
}

float IP64Fxu7Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT compact_code, size_t dim
) {
    const hn::ScalableTag<float> d;
    auto sum = hn::Zero(d);
    for (size_t i = 0; i < dim; i += 64) {
        const uint8_t* g1 = compact_code;
        const uint8_t* g2 = compact_code + 16;
        const uint8_t* g3 = compact_code + 32;
        const uint8_t* top_bit = compact_code + 48;
        float codes[64];
        for (size_t k = 0; k < 16; ++k) {
            codes[k] = static_cast<float>(g1[k] & 0x3FU);
            codes[16 + k] = static_cast<float>(g2[k] & 0x3FU);
            codes[32 + k] = static_cast<float>(g3[k] & 0x3FU);
            const uint8_t high = static_cast<uint8_t>(
                ((g1[k] >> 6) & 0x3U) | (((g2[k] >> 6) & 0x3U) << 2) |
                (((g3[k] >> 6) & 0x3U) << 4)
            );
            codes[48 + k] = static_cast<float>(high);
        }
        for (size_t c = 0; c < 64; ++c) {
            const uint8_t top = (top_bit[c % 8] >> (c / 8)) & 1U;
            codes[c] += static_cast<float>(top << 6);
        }
        sum = DotAccumulate(d, query + i, codes, 64, sum);
        compact_code += 56;
    }
    return hn::ReduceSum(d, sum);
}

// ip16_fxu8_avx512 widens codes via a single native _mm512_cvtepu8_epi32 +
// _mm512_cvtepi32_ps, with 4-way unrolling to hide FMA latency; a
// microbenchmark of the original scalar-per-lane static_cast version here
// (2M calls, 960-dim) showed it ~1.5x slower (59.7 ns/call vs 40.4 ns/call)
// — the same "scalar where a vectorized widening op exists" gap as
// warmup_ip_x0_q_512_highway's earlier popcount issue and MaskIpX0QImpl's
// bit extraction. This uses a two-step PromoteTo chain (u8 -> u16 -> i32,
// each a basic/universal Highway promotion) rather than a target-specific
// single-step u8->i32 op, so it stays correct on every Highway target
// without needing to verify each one exposes that exact combined op; on
// AVX-512 the compiler still typically folds the chain into the same native
// instructions.
float IP16Fxu8Impl(
    const float* HWY_RESTRICT query, const uint8_t* HWY_RESTRICT code, size_t dim
) {
    const hn::ScalableTag<int32_t> di32;
    const hn::RebindToFloat<decltype(di32)> d;
    const hn::Rebind<uint16_t, decltype(di32)> du16;
    const hn::Rebind<uint8_t, decltype(du16)> du8;
    const size_t lanes = hn::Lanes(d);

    auto widen_codes = [&](size_t at) {
        return hn::ConvertTo(
            d, hn::PromoteTo(di32, hn::PromoteTo(du16, hn::LoadU(du8, code + at)))
        );
    };

    auto s0 = hn::Zero(d);
    auto s1 = hn::Zero(d);
    auto s2 = hn::Zero(d);
    auto s3 = hn::Zero(d);
    size_t i = 0;
    for (; i + (lanes * 4) <= dim; i += lanes * 4) {
        s0 = hn::MulAdd(widen_codes(i), hn::LoadU(d, query + i), s0);
        s1 = hn::MulAdd(widen_codes(i + lanes), hn::LoadU(d, query + i + lanes), s1);
        s2 = hn::MulAdd(
            widen_codes(i + (lanes * 2)), hn::LoadU(d, query + i + (lanes * 2)), s2
        );
        s3 = hn::MulAdd(
            widen_codes(i + (lanes * 3)), hn::LoadU(d, query + i + (lanes * 3)), s3
        );
    }
    auto sum = hn::Add(hn::Add(s0, s1), hn::Add(s2, s3));
    for (; i + lanes <= dim; i += lanes) {
        sum = hn::MulAdd(widen_codes(i), hn::LoadU(d, query + i), sum);
    }
    float result = hn::ReduceSum(d, sum);
    for (; i < dim; ++i) {
        result += query[i] * static_cast<float>(code[i]);
    }
    return result;
}

}  // namespace HWY_NAMESPACE
}  // namespace rabitqlib::simd::excode_ipimpl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rabitqlib::simd::excode_ipimpl {

HWY_EXPORT(IP16Fxu1Impl);
HWY_EXPORT(IP64Fxu2Impl);
HWY_EXPORT(IP64Fxu3Impl);
HWY_EXPORT(IP16Fxu4Impl);
HWY_EXPORT(IP64Fxu5Impl);
HWY_EXPORT(IP64Fxu6Impl);
HWY_EXPORT(IP64Fxu7Impl);
HWY_EXPORT(IP16Fxu8Impl);

float ip16_fxu1_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP16Fxu1Impl)(query, compact_code, dim);
}

float ip64_fxu2_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP64Fxu2Impl)(query, compact_code, dim);
}

float ip64_fxu3_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP64Fxu3Impl)(query, compact_code, dim);
}

float ip16_fxu4_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP16Fxu4Impl)(query, compact_code, dim);
}

float ip64_fxu5_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP64Fxu5Impl)(query, compact_code, dim);
}

float ip64_fxu6_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP64Fxu6Impl)(query, compact_code, dim);
}

float ip64_fxu7_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP64Fxu7Impl)(query, compact_code, dim);
}

float ip16_fxu8_highway(
    const float* __restrict__ query, const uint8_t* __restrict__ code, size_t dim
) {
    return HWY_DYNAMIC_DISPATCH(IP16Fxu8Impl)(query, code, dim);
}

}  // namespace rabitqlib::simd::excode_ipimpl
#endif  // HWY_ONCE
