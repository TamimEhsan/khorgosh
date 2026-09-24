// Portable (Highway) FHT/Kac rotation kernels. See
// docs/portability/highway-plan.md Phase 3.
//
// fht_kernels.hpp's AVX intrinsics implement a Fast Walsh-Hadamard Transform
// via a radix-8 leaf (three fused intra-register butterfly stages) merged
// with wider inter-register butterflies. That radix-8 structure only exists
// for cache/register efficiency: a Walsh-Hadamard transform of size N is the
// tensor product H_2 (x) H_2 (x) ... (x) H_2 (log2(N) times), and the order
// in which those log2(N) butterfly "levels" are applied does not change the
// natural-order result. The plain iterative doubling algorithm below
// (radix-2, one level per loop iteration) is therefore not an approximation
// of the AVX version — it computes the same transform — and matches it
// within the existing FhtDispatchTest.BackendsMatchScalarButterfliesAndPadding
// tolerance (verified). It costs a modest constant factor in extra memory
// traffic (no fused multi-stage register reuse), traded here for being
// correct at any Highway vector width without replicating per-width
// intra-register lane-shuffle networks.

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/rotator_highway.cpp"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rabitqlib/simd/rotator_dispatch.hpp"

HWY_BEFORE_NAMESPACE();
namespace rabitqlib::simd {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Bit i of flip[i / 8] (LSB-first) selects the sign of data[i]; matches
// flip_sign_avx2's byte/bit-index convention exactly (see rotator_avx2.cpp).
// Scalar bit tests build a per-lane XOR mask; this is not the hottest of hot
// paths (rotation runs once per insert/query, not per candidate), so
// correctness and width-portability are prioritized over avoiding the
// scalar mask setup.
void FlipSignImpl(const uint8_t* HWY_RESTRICT flip, float* HWY_RESTRICT data, size_t dim) {
    const hn::ScalableTag<float> d;
    const hn::RebindToUnsigned<decltype(d)> du;
    const size_t lanes = hn::Lanes(d);
    constexpr uint32_t kSignBit = 0x80000000U;

    size_t i = 0;
    for (; i + lanes <= dim; i += lanes) {
        uint32_t mask_bits[hn::MaxLanes(d)];
        for (size_t lane = 0; lane < lanes; ++lane) {
            const size_t pos = i + lane;
            const bool flip_bit = ((flip[pos / 8] >> (pos % 8)) & 1U) != 0;
            mask_bits[lane] = flip_bit ? kSignBit : 0U;
        }
        const auto mask = hn::LoadU(du, mask_bits);
        const auto bits = hn::Xor(hn::BitCast(du, hn::LoadU(d, data + i)), mask);
        hn::StoreU(hn::BitCast(d, bits), d, data + i);
    }
    for (; i < dim; ++i) {
        if (((flip[i / 8] >> (i % 8)) & 1U) != 0) {
            data[i] = -data[i];
        }
    }
}

// new_x = x + y, new_y = x - y between the first and second half; matches
// kacs_walk_avx2. Requires len % 2 == 0 (guaranteed: len is always
// padded_dim, a multiple of 64).
void KacsWalkImpl(float* HWY_RESTRICT data, size_t len) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const size_t half = len / 2;
    float* HWY_RESTRICT lo = data;
    float* HWY_RESTRICT hi = data + half;

    size_t i = 0;
    for (; i + lanes <= half; i += lanes) {
        const auto x = hn::LoadU(d, lo + i);
        const auto y = hn::LoadU(d, hi + i);
        hn::StoreU(hn::Add(x, y), d, lo + i);
        hn::StoreU(hn::Sub(x, y), d, hi + i);
    }
    for (; i < half; ++i) {
        const float x = lo[i];
        const float y = hi[i];
        lo[i] = x + y;
        hi[i] = x - y;
    }
}

inline void Rescale(float* HWY_RESTRICT data, size_t dim, float factor) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto fac = hn::Set(d, factor);
    size_t i = 0;
    for (; i + lanes <= dim; i += lanes) {
        hn::StoreU(hn::Mul(hn::LoadU(d, data + i), fac), d, data + i);
    }
    for (; i < dim; ++i) {
        data[i] *= factor;
    }
}

// In-place iterative Walsh-Hadamard transform of data[0, n), n a power of
// two. See the file comment for why this radix-2 structure matches the
// AVX2/AVX-512 radix-8 kernels' natural-order output.
inline void Fwht(float* HWY_RESTRICT data, size_t n) {
    const hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    for (size_t width = 1; width < n; width *= 2) {
        if (width >= lanes) {
            for (size_t block = 0; block < n; block += width * 2) {
                float* HWY_RESTRICT lo = data + block;
                float* HWY_RESTRICT hi = lo + width;
                for (size_t i = 0; i < width; i += lanes) {
                    const auto a = hn::LoadU(d, lo + i);
                    const auto b = hn::LoadU(d, hi + i);
                    hn::StoreU(hn::Add(a, b), d, lo + i);
                    hn::StoreU(hn::Sub(a, b), d, hi + i);
                }
            }
        } else {
            for (size_t block = 0; block < n; block += width * 2) {
                for (size_t i = 0; i < width; ++i) {
                    const size_t x = block + i;
                    const size_t y = x + width;
                    const float a = data[x];
                    const float b = data[y];
                    data[x] = a + b;
                    data[y] = a - b;
                }
            }
        }
    }
}

// Mirrors fht_rotate_impl in rotator_kernels.hpp exactly (4 rounds of
// flip_sign + FHT + rescale, with kacs_walk mixing in the padded tail when
// trunc_dim < padded_dim), using the portable helpers above in place of the
// AVX2/AVX-512-templated ones.
void FhtRotateImpl(
    const float* HWY_RESTRICT data,
    float* HWY_RESTRICT rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* HWY_RESTRICT flip
) {
    std::memcpy(rotated_vec, data, sizeof(float) * dim);
    std::fill(rotated_vec + dim, rotated_vec + padded_dim, 0.0F);

    if (trunc_dim == padded_dim) {
        FlipSignImpl(flip, rotated_vec, padded_dim);
        Fwht(rotated_vec, trunc_dim);
        Rescale(rotated_vec, trunc_dim, fac);

        FlipSignImpl(flip + (padded_dim / 8), rotated_vec, padded_dim);
        Fwht(rotated_vec, trunc_dim);
        Rescale(rotated_vec, trunc_dim, fac);

        FlipSignImpl(flip + (2 * padded_dim / 8), rotated_vec, padded_dim);
        Fwht(rotated_vec, trunc_dim);
        Rescale(rotated_vec, trunc_dim, fac);

        FlipSignImpl(flip + (3 * padded_dim / 8), rotated_vec, padded_dim);
        Fwht(rotated_vec, trunc_dim);
        Rescale(rotated_vec, trunc_dim, fac);

        return;
    }

    const size_t start = padded_dim - trunc_dim;

    FlipSignImpl(flip, rotated_vec, padded_dim);
    Fwht(rotated_vec, trunc_dim);
    Rescale(rotated_vec, trunc_dim, fac);
    KacsWalkImpl(rotated_vec, padded_dim);

    FlipSignImpl(flip + (padded_dim / 8), rotated_vec, padded_dim);
    Fwht(rotated_vec + start, trunc_dim);
    Rescale(rotated_vec + start, trunc_dim, fac);
    KacsWalkImpl(rotated_vec, padded_dim);

    FlipSignImpl(flip + (2 * padded_dim / 8), rotated_vec, padded_dim);
    Fwht(rotated_vec, trunc_dim);
    Rescale(rotated_vec, trunc_dim, fac);
    KacsWalkImpl(rotated_vec, padded_dim);

    FlipSignImpl(flip + (3 * padded_dim / 8), rotated_vec, padded_dim);
    Fwht(rotated_vec + start, trunc_dim);
    Rescale(rotated_vec + start, trunc_dim, fac);
    KacsWalkImpl(rotated_vec, padded_dim);

    // This can be removed if we don't care about the absolute value of
    // similarities. Matches fht_rotate_impl.
    Rescale(rotated_vec, padded_dim, 0.25F);
}

}  // namespace HWY_NAMESPACE
}  // namespace rabitqlib::simd
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace rabitqlib::simd {

HWY_EXPORT(FlipSignImpl);
HWY_EXPORT(KacsWalkImpl);
HWY_EXPORT(FhtRotateImpl);

void flip_sign_highway(const uint8_t* flip, float* data, size_t dim) {
    HWY_DYNAMIC_DISPATCH(FlipSignImpl)(flip, data, dim);
}

void kacs_walk_highway(float* data, size_t len) {
    HWY_DYNAMIC_DISPATCH(KacsWalkImpl)(data, len);
}

void fht_rotate_highway(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
) {
    HWY_DYNAMIC_DISPATCH(FhtRotateImpl)
    (data, rotated_vec, dim, padded_dim, trunc_dim, fac, flip);
}

}  // namespace rabitqlib::simd
#endif  // HWY_ONCE
