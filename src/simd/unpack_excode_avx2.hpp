#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rabitqlib/quantization/unpack_excode.hpp"

// AVX2 form of the 64-dimension unpackers: two __m256i, low holding dimensions
// 0-31 and high holding 32-63. Inline for the same reason as the AVX-512 twin.
//
// The packed layouts are ISA-independent -- pack_excode_avx2.cpp and
// pack_excode_avx512.cpp forward to the same intrinsics, and those use only
// 128-bit SSE ops. So the inverses need nothing exotic either: shift-and-mask
// within 16-bit lanes, plus a shuffle-and-compare for the 1-bit plane. Widths
// 3, 5, 6 and 7 are still on the scalar interior; their transposed top-bit
// plane is the part that needs real work.

namespace rabitqlib::simd {

namespace detail {

// 16 dims from one uint16: spread its two bytes across 16 lanes, isolate one bit
// per lane, then turn "nonzero" into exactly 1.
inline __m128i unpack16_1bit_sse(const uint8_t* in) {
    const __m128i spread_bytes =
        _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    const __m128i bit_of_lane = _mm_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128
    );
    uint16_t code;
    std::memcpy(&code, in, sizeof(code));
    __m128i v = _mm_shuffle_epi8(_mm_set1_epi16(static_cast<short>(code)), spread_bytes);
    __m128i isolated = _mm_and_si128(v, bit_of_lane);
    return _mm_and_si128(_mm_cmpeq_epi8(isolated, bit_of_lane), _mm_set1_epi8(1));
}

}  // namespace detail

inline void unpack64_1bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    lo = _mm256_set_m128i(detail::unpack16_1bit_sse(in + 2), detail::unpack16_1bit_sse(in));
    hi = _mm256_set_m128i(
        detail::unpack16_1bit_sse(in + 6), detail::unpack16_1bit_sse(in + 4)
    );
}

inline void unpack64_2bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    // 16 bytes hold all 64 dims: byte k carries dim k at bits 0-1, k+16 at 2-3,
    // k+32 at 4-5, k+48 at 6-7. The 16-bit shift drags in neighbouring bits from
    // above, which the 2-bit mask drops.
    const __m128i m = _mm_set1_epi8(0x3);
    __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in));
    __m128i g0 = _mm_and_si128(c, m);
    __m128i g1 = _mm_and_si128(_mm_srli_epi16(c, 2), m);
    __m128i g2 = _mm_and_si128(_mm_srli_epi16(c, 4), m);
    __m128i g3 = _mm_and_si128(_mm_srli_epi16(c, 6), m);
    lo = _mm256_set_m128i(g1, g0);
    hi = _mm256_set_m128i(g3, g2);
}

inline void unpack64_3bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    alignas(32) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 3);
    lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw));
    hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw + 32));
}

namespace detail {

// One 16-dim group: 8 bytes, byte k holding dim k low and dim k+8 high. The
// output is the two nibble sets concatenated, not interleaved, which is exactly
// what unpacklo_epi64 gives.
inline __m128i unpack16_4bit_sse(const uint8_t* in) {
    const __m128i m = _mm_set1_epi8(0xF);
    __m128i v = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(in));
    __m128i lo_n = _mm_and_si128(v, m);
    __m128i hi_n = _mm_and_si128(_mm_srli_epi16(v, 4), m);
    return _mm_unpacklo_epi64(lo_n, hi_n);
}

}  // namespace detail

inline void unpack64_4bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    lo = _mm256_set_m128i(detail::unpack16_4bit_sse(in + 8), detail::unpack16_4bit_sse(in));
    hi = _mm256_set_m128i(
        detail::unpack16_4bit_sse(in + 24), detail::unpack16_4bit_sse(in + 16)
    );
}

inline void unpack64_5bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    alignas(32) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 5);
    lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw));
    hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw + 32));
}

inline void unpack64_6bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    alignas(32) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 6);
    lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw));
    hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw + 32));
}

inline void unpack64_7bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    alignas(32) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 7);
    lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw));
    hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(raw + 32));
}

inline void unpack64_8bit_excode_avx2(const uint8_t* in, __m256i& lo, __m256i& hi) {
    lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in));
    hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + 32));
}

inline void unpack64_excode_avx2(const uint8_t* in, size_t bits, __m256i& lo, __m256i& hi) {
    switch (bits) {
        case 1: unpack64_1bit_excode_avx2(in, lo, hi); break;
        case 2: unpack64_2bit_excode_avx2(in, lo, hi); break;
        case 3: unpack64_3bit_excode_avx2(in, lo, hi); break;
        case 4: unpack64_4bit_excode_avx2(in, lo, hi); break;
        case 5: unpack64_5bit_excode_avx2(in, lo, hi); break;
        case 6: unpack64_6bit_excode_avx2(in, lo, hi); break;
        case 7: unpack64_7bit_excode_avx2(in, lo, hi); break;
        default: unpack64_8bit_excode_avx2(in, lo, hi); break;
    }
}

}  // namespace rabitqlib::simd
