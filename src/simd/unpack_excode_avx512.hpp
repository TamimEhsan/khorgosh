#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "rabitqlib/quantization/unpack_excode.hpp"

// Unpack exactly 64 dimensions of packed ex-code into one byte per dimension,
// delivered in a __m512i. These are inline on purpose: this runs in the
// innermost search loop, so a cross-TU call would cost more than the unpack.
// (The packers next door are compiled units because packing happens once per
// vector at index build, where a call is free -- do not copy that layout here.)
//
// Each width has its own entry point so it can be specialised independently.
// Widths without a hand-written form fall back to the scalar unpacker, which is
// correct but byte-at-a-time; the scalar version is also the oracle the
// specialised ones are tested against.

namespace rabitqlib::simd {

inline void unpack64_1bit_excode_avx512(const uint8_t* in, __m512i& out) {
    // 64 dims are exactly a 64-bit mask, bit k == dim k
    uint64_t bits;
    std::memcpy(&bits, in, sizeof(bits));
    out = _mm512_maskz_mov_epi8(static_cast<__mmask64>(bits), _mm512_set1_epi8(1));
}

inline void unpack64_2bit_excode_avx512(const uint8_t* in, __m512i& out) {
    // 16 bytes hold all 64 dims. Broadcasting them into the four 128-bit lanes
    // puts each lane's shift on its own dim range: 0-15 shift 0, 16-31 shift 2,
    // 32-47 shift 4, 48-63 shift 6. The 32-bit shift drags neighbouring bits in
    // from above, which the 2-bit mask then drops.
    const __m512i shift = _mm512_setr_epi32(0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 6, 6, 6, 6);
    __m512i b =
        _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in)));
    out = _mm512_and_si512(_mm512_srlv_epi32(b, shift), _mm512_set1_epi8(0x3));
}

inline void unpack64_3bit_excode_avx512(const uint8_t* in, __m512i& out) {
    alignas(64) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 3);
    out = _mm512_load_si512(reinterpret_cast<const __m512i*>(raw));
}

inline void unpack64_4bit_excode_avx512(const uint8_t* in, __m512i& out) {
    // Groups of 16 dims live in 8 bytes: byte k holds dim k in its low nibble
    // and dim k+8 in its high nibble. Each group must expand to 16 output bytes
    // as [lows(0..7), highs(0..7)], not interleaved -- so broadcast each qword
    // into both halves of its 128-bit lane and blend by byte position.
    const __m512i mask4 = _mm512_set1_epi8(0xF);
    __m512i q = _mm512_castsi256_si512(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in))
    );
    q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3), q);
    __m512i lo = _mm512_and_si512(q, mask4);
    __m512i hi = _mm512_and_si512(_mm512_srli_epi64(q, 4), mask4);
    out = _mm512_mask_blend_epi8(static_cast<__mmask64>(0xFF00FF00FF00FF00ULL), lo, hi);
}

inline void unpack64_5bit_excode_avx512(const uint8_t* in, __m512i& out) {
    alignas(64) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 5);
    out = _mm512_load_si512(reinterpret_cast<const __m512i*>(raw));
}

inline void unpack64_6bit_excode_avx512(const uint8_t* in, __m512i& out) {
    alignas(64) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 6);
    out = _mm512_load_si512(reinterpret_cast<const __m512i*>(raw));
}

inline void unpack64_7bit_excode_avx512(const uint8_t* in, __m512i& out) {
    alignas(64) uint8_t raw[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(in, raw, 64, 7);
    out = _mm512_load_si512(reinterpret_cast<const __m512i*>(raw));
}

inline void unpack64_8bit_excode_avx512(const uint8_t* in, __m512i& out) {
    out = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(in));
}

inline void unpack64_excode_avx512(const uint8_t* in, size_t bits, __m512i& out) {
    switch (bits) {
        case 1: unpack64_1bit_excode_avx512(in, out); break;
        case 2: unpack64_2bit_excode_avx512(in, out); break;
        case 3: unpack64_3bit_excode_avx512(in, out); break;
        case 4: unpack64_4bit_excode_avx512(in, out); break;
        case 5: unpack64_5bit_excode_avx512(in, out); break;
        case 6: unpack64_6bit_excode_avx512(in, out); break;
        case 7: unpack64_7bit_excode_avx512(in, out); break;
        default: unpack64_8bit_excode_avx512(in, out); break;
    }
}

}  // namespace rabitqlib::simd
