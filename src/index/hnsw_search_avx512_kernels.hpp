#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

#include "rabitqlib/index/query.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"

inline int early_exit_count[20];


namespace rabitqlib::hnsw::detail {

static inline float hnsw_mask_ip_x0_q_avx512(
    const float* query, const uint64_t* data, size_t padded_dim
) {
    const size_t num_blk = padded_dim / 64;
    const uint64_t* it_data = data;
    const float* it_query = query;

    //    __m512 sum0 = _mm512_setzero_ps();
    //    __m512 sum1 = _mm512_setzero_ps();
    //    __m512 sum2 = _mm512_setzero_ps();
    //    __m512 sum3 = _mm512_setzero_ps();

    __m512 sum = _mm512_setzero_ps();
    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t bits = rabitqlib::reverse_bits_u64(*it_data);

        auto mask0 = static_cast<__mmask16>(bits);
        auto mask1 = static_cast<__mmask16>(bits >> 16);
        auto mask2 = static_cast<__mmask16>(bits >> 32);
        auto mask3 = static_cast<__mmask16>(bits >> 48);

        __m512 masked0 = _mm512_maskz_loadu_ps(mask0, it_query);
        __m512 masked1 = _mm512_maskz_loadu_ps(mask1, it_query + 16);
        __m512 masked2 = _mm512_maskz_loadu_ps(mask2, it_query + 32);
        __m512 masked3 = _mm512_maskz_loadu_ps(mask3, it_query + 48);

        sum = _mm512_add_ps(sum, masked0);
        sum = _mm512_add_ps(sum, masked1);
        sum = _mm512_add_ps(sum, masked2);
        sum = _mm512_add_ps(sum, masked3);

        //         _mm_prefetch(reinterpret_cast<const char*>(it_query + 128), _MM_HINT_T1);

        ++it_data;
        it_query += 64;
    }

    //    __m512 sum = _mm512_add_ps(_mm512_add_ps(sum0, sum1), _mm512_add_ps(sum2, sum3));
    return _mm512_reduce_add_ps(sum);
}

template <size_t b_query>
static inline float hnsw_warmup_ip_x0_q_512_avx512_1_7(
    const uint64_t* __restrict data,
    const uint64_t* __restrict query,
    float delta,
    float vl,
    size_t padded_dim
) {
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    __m512i acc_ip = _mm512_setzero_si512();
    __m512i acc_ppc = _mm512_setzero_si512();

    size_t i = 0;
    size_t dim_end_512 = (padded_dim / 512) * 512;

    __m512i acc_bits[SplitSingleQuery<float>::kNumBits];
    for (size_t j = 0; j < b_query; ++j) {
        acc_bits[j] = _mm512_setzero_si512();
    }

    for (; i < dim_end_512; i += 512) {
        __m512i data_vec = _mm512_loadu_si512(data);
        data += 8;

        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_popcnt_epi64(data_vec));

        for (size_t j = 0; j < b_query; ++j) {
            __m512i query_vec = _mm512_loadu_si512(query);
            query += 8;

            __m512i pop = _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec));
            acc_bits[j] = _mm512_add_epi64(acc_bits[j], pop);
        }
    }

    size_t remaining_dim = padded_dim - i;
    if (remaining_dim > 0) {
        size_t num_chunks = remaining_dim / 64;
        auto valid_mask = static_cast<__mmask8>((1u << num_chunks) - 1u);

        __m512i data_vec = _mm512_maskz_loadu_epi64(valid_mask, data);
        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_popcnt_epi64(data_vec));

        for (size_t j = 0; j < b_query; ++j) {
            __m512i query_vec = _mm512_maskz_loadu_epi64(valid_mask, query);
            query += num_chunks;

            __m512i pop = _mm512_popcnt_epi64(_mm512_and_si512(data_vec, query_vec));
            acc_bits[j] = _mm512_add_epi64(acc_bits[j], pop);
        }
    }

    for (size_t j = 0; j < b_query; ++j) {
        __m128i shift = _mm_cvtsi32_si128(static_cast<int>(j));
        acc_ip = _mm512_add_epi64(acc_ip, _mm512_sll_epi64(acc_bits[j], shift));
    }

    ip_scalar += static_cast<size_t>(_mm512_reduce_add_epi64(acc_ip));
    ppc_scalar += static_cast<size_t>(_mm512_reduce_add_epi64(acc_ppc));

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}

static inline float hnsw_warmup_ip_x0_q_512_avx512_8_fast(
    const uint64_t* __restrict data,
    const uint64_t* __restrict query64,
    float delta,
    float vl,
    size_t padded_dim
) {
    auto query = reinterpret_cast<const uint8_t*>(query64);

    // 1. Multiple accumulators to break the loop-carried dependency chain
    __m512i acc_ip0 = _mm512_setzero_si512();
    __m512i acc_ip1 = _mm512_setzero_si512();
    __m512i acc_ip2 = _mm512_setzero_si512();
    __m512i acc_ip3 = _mm512_setzero_si512();

    __m512i acc_ppc = _mm512_setzero_si512();
    const __m512i vzero = _mm512_setzero_si512();

    size_t i = 0;
    size_t dim_end_512 = (padded_dim / 512) * 512;

    // Process 512 dimensions per iteration (8 x 64-bit blocks)
    for (; i < dim_end_512; i += 512) {
        // 2. Vectorize the PPC count across all 512 database bits at once
        __m512i db_vec = _mm512_loadu_si512(data);
        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_popcnt_epi64(db_vec));

        // 3. Extract masks into opmask registers directly from memory
        __mmask64 m0 = _cvtu64_mask64(data[0]);
        __mmask64 m1 = _cvtu64_mask64(data[1]);
        __mmask64 m2 = _cvtu64_mask64(data[2]);
        __mmask64 m3 = _cvtu64_mask64(data[3]);
        __mmask64 m4 = _cvtu64_mask64(data[4]);
        __mmask64 m5 = _cvtu64_mask64(data[5]);
        __mmask64 m6 = _cvtu64_mask64(data[6]);
        __mmask64 m7 = _cvtu64_mask64(data[7]);
        data += 8;

        // 4. Fused Masked Load: Emits single 'vmovdqu8 zmm {k}{z}, [mem]' instructions
        __m512i q0 = _mm512_maskz_loadu_epi8(m0, query + i + 0);
        __m512i q1 = _mm512_maskz_loadu_epi8(m1, query + i + 64);
        __m512i q2 = _mm512_maskz_loadu_epi8(m2, query + i + 128);
        __m512i q3 = _mm512_maskz_loadu_epi8(m3, query + i + 192);

        __m512i q4 = _mm512_maskz_loadu_epi8(m4, query + i + 256);
        __m512i q5 = _mm512_maskz_loadu_epi8(m5, query + i + 320);
        __m512i q6 = _mm512_maskz_loadu_epi8(m6, query + i + 384);
        __m512i q7 = _mm512_maskz_loadu_epi8(m7, query + i + 448);

        // Interleave SAD and addition across 4 independent stream chains
        acc_ip0 = _mm512_add_epi64(acc_ip0, _mm512_sad_epu8(q0, vzero));
        acc_ip1 = _mm512_add_epi64(acc_ip1, _mm512_sad_epu8(q1, vzero));
        acc_ip2 = _mm512_add_epi64(acc_ip2, _mm512_sad_epu8(q2, vzero));
        acc_ip3 = _mm512_add_epi64(acc_ip3, _mm512_sad_epu8(q3, vzero));

        acc_ip0 = _mm512_add_epi64(acc_ip0, _mm512_sad_epu8(q4, vzero));
        acc_ip1 = _mm512_add_epi64(acc_ip1, _mm512_sad_epu8(q5, vzero));
        acc_ip2 = _mm512_add_epi64(acc_ip2, _mm512_sad_epu8(q6, vzero));
        acc_ip3 = _mm512_add_epi64(acc_ip3, _mm512_sad_epu8(q7, vzero));
    }

    // Combine accumulators
    __m512i acc_ip = _mm512_add_epi64(
        _mm512_add_epi64(acc_ip0, acc_ip1),
        _mm512_add_epi64(acc_ip2, acc_ip3)
    );

    // Tail handling for dimensions not aligned to 512
    size_t dim_end_64 = (padded_dim / 64) * 64;
    for (; i < dim_end_64; i += 64) {
        __mmask64 m = _cvtu64_mask64(*data);
        acc_ppc = _mm512_add_epi64(acc_ppc, _mm512_set_epi64(0,0,0,0,0,0,0, _mm_popcnt_u64(*data)));
        data++;

        __m512i q = _mm512_maskz_loadu_epi8(m, query + i);
        acc_ip = _mm512_add_epi64(acc_ip, _mm512_sad_epu8(q, vzero));
    }

    size_t ip_scalar = static_cast<size_t>(_mm512_reduce_add_epi64(acc_ip));
    size_t ppc_scalar = static_cast<size_t>(_mm512_reduce_add_epi64(acc_ppc));

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
}


static inline float hnsw_warmup_ip_x0_q_512_avx512_8(
    const uint64_t* __restrict data,
    const uint64_t* __restrict query64,
    float delta,
    float target,
    size_t padded_dim
) {
    // std::cout << "Meo meo 2 " << delta << " " << vl << std::endl;
    // exit(1);
    auto query = reinterpret_cast<const uint8_t*>(query64);
    __m512i acc_ip = _mm512_setzero_si512();

    size_t i = 0;
    // Process 64 dimensions per loop iteration
    size_t dim_end_64 = (padded_dim / 64) * 64;

    for (; i < dim_end_64; i += 64) {
        // 1. Load 64 dimensions of the 1-bit database to use directly as a mask
        __mmask64 db_mask = *data;
        data++;

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
        float ip_scalar = static_cast<float>(_mm512_reduce_add_epi64(acc_ip));
        if (delta * ip_scalar > target) {
            // std::cout << "Early exit in warmup_ip_x0_q_512_avx512_8: delta * ip_scalar = " << delta * ip_scalar << " > target = " << target << std::endl;
            // exit(1);
            early_exit_count[i / 64]++;
            return (delta * ip_scalar);
        }
    }

    // Reduce the eight 64-bit accumulators down to a single scalar
    size_t ip_scalar = static_cast<size_t>(_mm512_reduce_add_epi64(acc_ip));

    return (delta * static_cast<float>(ip_scalar));
}

static inline float hnsw_warmup_ip_x0_q_512_avx512(
    const uint64_t* __restrict data,
    const uint64_t* __restrict query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    switch (b_query) {
        case 1:
            return hnsw_warmup_ip_x0_q_512_avx512_1_7<1>(data, query, delta, vl, padded_dim);
        case 2:
            return hnsw_warmup_ip_x0_q_512_avx512_1_7<2>(data, query, delta, vl, padded_dim);
        case 4:
            return hnsw_warmup_ip_x0_q_512_avx512_1_7<4>(data, query, delta, vl, padded_dim);
        case 8:
            // return hnsw_warmup_ip_x0_q_512_avx512_1_7<8>(data, query, delta, vl, padded_dim);
            return hnsw_warmup_ip_x0_q_512_avx512_8(data, query, delta, vl, padded_dim);
        default:
            throw std::invalid_argument("Unsupported number of queries for AVX-512 warmup kernel");
    }
}

static inline uint64_t reverse_bits_u64(uint64_t n) {
    n = ((n >> 1) & 0x5555555555555555) | ((n << 1) & 0xaaaaaaaaaaaaaaaa);
    n = ((n >> 2) & 0x3333333333333333) | ((n << 2) & 0xcccccccccccccccc);
    n = ((n >> 4) & 0x0f0f0f0f0f0f0f0f) | ((n << 4) & 0xf0f0f0f0f0f0f0f0);
    n = ((n >> 8) & 0x00ff00ff00ff00ff) | ((n << 8) & 0xff00ff00ff00ff00);
    n = ((n >> 16) & 0x0000ffff0000ffff) | ((n << 16) & 0xffff0000ffff0000);
    n = ((n >> 32) & 0x00000000ffffffff) | ((n << 32) & 0xffffffff00000000);
    return n;
}


}  // namespace rabitqlib::hnsw::detail
