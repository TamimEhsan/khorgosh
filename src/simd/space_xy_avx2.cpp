#include "rabitqlib/simd/space_xy_dispatch.hpp"

#include <immintrin.h>

#include "unpack_excode_avx2.hpp"

namespace rabitqlib::simd {

namespace {

inline int32_t hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi32(lo, hi);
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(1, 0, 3, 2)));
    s = _mm_add_epi32(s, _mm_shuffle_epi32(s, _MM_SHUFFLE(2, 3, 0, 1)));
    return _mm_cvtsi128_si32(s);
}

inline int64_t hsum_epi64(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i s = _mm_add_epi64(lo, hi);
    return _mm_cvtsi128_si64(s) + _mm_extract_epi64(s, 1);
}

// 32 dimensions of code against 32 of query, accumulating both terms.
inline void accumulate32(
    __m256i codes, const uint8_t* query, __m256i& acc_ip, __m256i& acc_sum
) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query));
    // maddubs treats the first operand as unsigned and the second as signed,
    // matching dpbusd's convention, so the code goes first here too.
    __m256i pairs = _mm256_maddubs_epi16(codes, q);
    acc_ip = _mm256_add_epi32(acc_ip, _mm256_madd_epi16(pairs, ones16));
    acc_sum = _mm256_add_epi64(acc_sum, _mm256_sad_epu8(codes, _mm256_setzero_si256()));
}

}  // namespace

// AVX2 form of the quantized-query base inner product, for machines with AVX2
// but without AVX512-VNNI. Same contract as the VNNI kernel.
//
// Without VNNI the dot product goes through _mm256_maddubs_epi16, which sums
// two byte products into a SIGNED 16-bit lane and saturates. That bounds the
// inputs: 2 * (2^base_bits - 1) * (2^query_bits - 1) must stay under 32767.
// At the current query width of 4 bits the worst case is 2*255*15 = 7650, so
// every base_bits is safe; a query wider than 6 bits would not be, and would
// need the products widened before summing.
float xy_base_ip_u8_avx2(
    const uint8_t* quant_query,
    const uint8_t* code,
    size_t dim,
    size_t base_bits,
    float delta,
    float vl
) {
    __m256i acc_ip = _mm256_setzero_si256();
    __m256i acc_sum = _mm256_setzero_si256();

    const size_t bytes_per_block = base_bits * 8;  // 64 dims worth of packed code

    for (size_t d = 0; d < dim; d += 64) {
        __m256i lo;
        __m256i hi;
        unpack64_excode_avx2(code + ((d / 64) * bytes_per_block), base_bits, lo, hi);
        accumulate32(lo, quant_query + d, acc_ip, acc_sum);
        accumulate32(hi, quant_query + d + 32, acc_ip, acc_sum);
    }

    auto ip = static_cast<float>(hsum_epi32(acc_ip));
    auto sum = static_cast<float>(hsum_epi64(acc_sum));
    return (delta * ip) + (vl * sum);
}

}  // namespace rabitqlib::simd
