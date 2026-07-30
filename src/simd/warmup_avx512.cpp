#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace rabitqlib::simd {
static inline uint64_t reverse_bits_u64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555) | ((n << 1) & 0xaaaaaaaaaaaaaaaa);
    n = ((n >> 2) & 0x3333333333333333) | ((n << 2) & 0xcccccccccccccccc);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0f) | ((n << 4) & 0xf0f0f0f0f0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff00ff00ff) | ((n << 8) & 0xff00ff00ff00ff00);
    n = ((n >> 16) & 0x0000ffff0000ffff) | ((n << 16) & 0xffff0000ffff0000);
    n = ((n >> 32) & 0x00000000ffffffff) | ((n << 32) & 0xffffffff00000000);
    return n;
}
float warmup_ip_x0_q_512_avx512(
    const uint64_t* data,
    const uint64_t* query64,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    std::cout << "Meo meo 1 " << delta << " " << vl << std::endl;
    exit(1);
    auto query = reinterpret_cast<const uint8_t*>(query64);
    size_t ppc_scalar = 0;
    __m512i acc_ip = _mm512_setzero_si512();

    size_t i = 0;
    // Process 64 dimensions per loop iteration
    size_t dim_end_64 = (padded_dim / 64) * 64;

    for (; i < dim_end_64; i += 64) {
        // 1. Load 64 dimensions of the 1-bit database to use directly as a mask
        __mmask64 db_mask = *data;
        data++;

        // Track data popcnt for the 'vl' scaling factor later
        ppc_scalar += _mm_popcnt_u64(db_mask);

        // 2. Load 64 dimensions (64 bytes) of the 8-bit query
        __m512i query_vec = _mm512_loadu_si512((const __m512i*)(query + i));

        // 3. Mask out the query bytes where the DB bit is 0
        // Result: Bytes are kept if DB=1, zeroed out if DB=0
        __m512i masked_query = _mm512_maskz_mov_epi8(db_mask, query_vec);

        // 4. Fast horizontal sum of 8-bit integers into 64-bit accumulators
        // _mm512_sad_epu8 computes the Sum of Absolute Differences against zero,
        // which efficiently sums 8 adjacent bytes into a 64-bit integer.
        // It outputs eight 64-bit sums per vector.
        __m512i byte_sums = _mm512_sad_epu8(masked_query, _mm512_setzero_si512());

        // 5. Add to the master 64-bit accumulators
        acc_ip = _mm512_add_epi64(acc_ip, byte_sums);
    }

    // Reduce the eight 64-bit accumulators down to a single scalar
    size_t ip_scalar = static_cast<size_t>(_mm512_reduce_add_epi64(acc_ip));

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));

    // std::cout << "ip_scalar: " << ip_scalar << ", ppc_scalar: " << ppc_scalar << std::endl;
    // std::cout << (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar)) << std::endl;
    // return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));

}


// float warmup_ip_x0_q_512_avx512(
//     const uint64_t* data,
//     const uint64_t* query,
//     float delta,
//     float vl,
//     size_t padded_dim,
//     size_t b_query
// ) {
//     std::cout<< "Meo meo 2 " <<delta<<" " <<vl<< std::endl;
//     auto query_8 = reinterpret_cast<const uint8_t*>(query);
//     const size_t num_blk = padded_dim / 64;
//     const uint64_t* it_data = data;
//     const uint8_t* it_query = query_8;

//     // Accumulate sums in 32-bit integer SIMD registers to prevent overflow
//     __m512i sum_i32 = _mm512_setzero_si512();

//     for (size_t i = 0; i < num_blk; ++i) {
//         uint64_t bits = reverse_bits_u64(*it_data);
//         __mmask64 mask = static_cast<__mmask64>(bits);
//         // set mask to all 1
//         // mask = 0xFFFFFFFFFFFFFFFF;

//         // Masked load 64 uint8 elements directly using byte-level mask
//         __m512i v_u8 = _mm512_maskz_loadu_epi8(mask, it_query);

//         // Zero-extend 8-bit ints into 32-bit ints (4 vector registers of 16 ints each)
//         // Sad (Sum of Absolute Differences) against zero is a fast hardware trick for horizontal u8 sums
//         __m512i sad = _mm512_sad_epu8(v_u8, _mm512_setzero_si512());
        
//         // Combine the 64-bit SAD sums into our 32-bit accumulator
//         sum_i32 = _mm512_add_epi64(sum_i32, sad);

//         ++it_data;
//         it_query += 64;
//     }

//     // Horizontal reduction of 64-bit elements and convert final sum to float
//     int64_t total_sum = _mm512_reduce_add_epi64(sum_i32);
//     std::cout<< "Meo meo 3 " <<total_sum<< std::endl;
//     return static_cast<float>(total_sum);

// }

}  // namespace rabitqlib::simd
