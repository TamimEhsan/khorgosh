#include "rabitqlib/simd/space_xy_dispatch.hpp"

#include <immintrin.h>

#include <cstring>

#include "unpack_excode_avx512.hpp"

namespace rabitqlib::simd {

// 64 dimensions per iteration. The codes are unpacked to one byte each, then
// _mm512_dpbusd_epi32 accumulates sum(code * q_int) while _mm512_sad_epu8
// accumulates sum(code) from the same bytes.
//
// The operand order of dpbusd is not symmetric: the first vector operand is
// treated as UNSIGNED bytes and the second as SIGNED. The code goes in the
// unsigned slot, since base_bits == 8 reaches 255; the query goes in the signed
// slot, which is safe while the query's own width stays <= 7 bits.
float xy_base_ip_u8_avx512vnni(
    const uint8_t* quant_query,
    const uint8_t* code,
    size_t dim,
    size_t base_bits,
    float delta,
    float vl
) {
    const __m512i zero = _mm512_setzero_si512();
    __m512i acc_ip = _mm512_setzero_si512();
    __m512i acc_sum = _mm512_setzero_si512();

    const size_t bytes_per_block = base_bits * 8;  // 64 dims worth of packed code

    for (size_t d = 0; d < dim; d += 64) {
        const uint8_t* blk = code + ((d / 64) * bytes_per_block);
        __m512i codes;
        unpack64_excode_avx512(blk, base_bits, codes);

        __m512i query =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(quant_query + d));

        acc_ip = _mm512_dpbusd_epi32(acc_ip, codes, query);
        acc_sum = _mm512_add_epi64(acc_sum, _mm512_sad_epu8(codes, zero));
    }

    auto ip = static_cast<float>(_mm512_reduce_add_epi32(acc_ip));
    auto sum = static_cast<float>(_mm512_reduce_add_epi64(acc_sum));
    return (delta * ip) + (vl * sum);
}

}  // namespace rabitqlib::simd
