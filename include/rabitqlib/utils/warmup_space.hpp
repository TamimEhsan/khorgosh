#pragma once

#include <immintrin.h>

#include <cstddef>
#include <cstdint>

inline int early_exit_count[20] = {0};  // Global array to track early exits per block

// Helper: AVX2 64-bit Popcount; Mula's method
inline __m256i popcount_avx2(__m256i v) {
#if defined(__AVX2__)
    // Lookup table for population count of 0-15
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0f);

    // Count low nibbles
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i cnt_lo = _mm256_shuffle_epi8(lookup, lo);

    // Count high nibbles
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    __m256i cnt_hi = _mm256_shuffle_epi8(lookup, hi);

    // Add counts (bytes)
    __m256i cnt_bytes = _mm256_add_epi8(cnt_lo, cnt_hi);

    // Sum bytes horizontally into 64-bit integers (SAD against 0)
    return _mm256_sad_epu8(cnt_bytes, _mm256_setzero_si256());
#else
    std::cerr << "AVX2 is required for popcount_avx2\n";
    exit(1);
#endif
}

inline float warmup_ip_x0_q_512(
    const uint64_t* data,
    const uint64_t* query64,
    float delta,
    float vl,
    size_t padded_dim,
    [[maybe_unused]] size_t b_query,
    float threshold = 1000000.0
) {
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512BW__)
    auto query = reinterpret_cast<const uint8_t*>(query64);
    __m512i acc_ip = _mm512_setzero_si512();

    size_t i = 0;
    // Process 64 dimensions per loop iteration
    size_t dim_end_64 = (padded_dim / 64) * 64;

    for (; i < dim_end_64; i += 64) {
        // 1. Load 64 dimensions of the 1-bit database to use directly as a mask
        __mmask64 db_mask = *data;
        db_mask = ~db_mask; // Inverts all 64 bits
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

        if( i%192 or i+64 >= padded_dim ) continue;
        float ip_scalar = static_cast<float>(_mm512_reduce_add_epi64(acc_ip));
        ip_scalar = ip_scalar * padded_dim / (i + 64);
        float dist = delta * ip_scalar + vl;
        // dist = dist * std::sqrt(static_cast<float>(padded_dim)/(i+64));

        float epsilon0 = 20; // Example value for epsilon0
        double rejectThreshold = (1.0 + (epsilon0 / std::sqrt(i+64))) * threshold;
        if(dist > rejectThreshold and i+64 < padded_dim) {
            // early_exit_count[i/64]++;
            return dist;
        }
    }

    // Reduce the eight 64-bit accumulators down to a single scalar
    float ip_scalar = static_cast<float>(_mm512_reduce_add_epi64(acc_ip));

    return delta * ip_scalar + vl;
#elif defined(__AVX2__)
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    __m256i acc_ip = _mm256_setzero_si256();
    __m256i acc_ppc = _mm256_setzero_si256();

    size_t i = 0;
    // Step by 512 bits at a time (64 bytes = 16 elements of 32-bit integers)
    size_t dim_end_512 = (padded_dim / 512) * 512;

    __m256i acc_bits[b_query];
    for (size_t j = 0; j < b_query; ++j) {
        acc_bits[j] = _mm256_setzero_si256();
    }

    for (; i < dim_end_512; i += 512) {
        // Load 64 bytes of data using paired 32-byte loads
        __m256i data_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data));
        __m256i data_vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + 4));
        data += 8; // Advance 8 x 64-bit ints (64 bytes)

        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_lo));
        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_hi));

        for (size_t j = 0; j < b_query; ++j) {
            // Load 64 bytes of transposed query matching the 512-bit block layout
            __m256i query_vec_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query64));
            __m256i query_vec_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query64 + 4));
            query64 += 8; // Advance 8 x 64-bit ints (64 bytes)

            __m256i pop_lo = popcount_avx2(_mm256_and_si256(data_vec_lo, query_vec_lo));
            __m256i pop_hi = popcount_avx2(_mm256_and_si256(data_vec_hi, query_vec_hi));
            
            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_lo);
            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_hi);
        }
    }

    // Remainder block: handles leftovers less than 512 bits wide (e.g., last 448 bits)
    size_t remaining_dim = padded_dim - i;
    if (remaining_dim > 0) {
        size_t num_chunks_64 = remaining_dim / 64;
        size_t num_chunks_32 = remaining_dim / 32;

        size_t chunks_lo = (num_chunks_32 > 8) ? 8 : num_chunks_32;
        size_t chunks_hi = (num_chunks_32 > 8) ? (num_chunks_32 - 8) : 0;

        // 1. Create a baseline sequence register
        __m256i sequence = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

        // 2. Generate masks in-register using Greater-Than comparisons
        // If chunks_lo is 3, limit will be [3,3,3,3,3,3,3,3]. 
        // 3 > seq results in [-1, -1, -1, 0, 0, 0, 0, 0], which is the exact mask needed.
        __m256i limit_lo = _mm256_set1_epi32(static_cast<int>(chunks_lo));
        __m256i mask_lo  = _mm256_cmpgt_epi32(limit_lo, sequence);

        __m256i limit_hi = _mm256_set1_epi32(static_cast<int>(chunks_hi));
        __m256i mask_hi  = _mm256_cmpgt_epi32(limit_hi, sequence);

        // 3. Vectorized execution continues with zero memory latency
        __m256i data_vec_lo = _mm256_maskload_epi32(reinterpret_cast<const int*>(data), mask_lo);
        __m256i data_vec_hi = _mm256_maskload_epi32(reinterpret_cast<const int*>(data + 4), mask_hi);
        
        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_lo));
        acc_ppc = _mm256_add_epi64(acc_ppc, popcount_avx2(data_vec_hi));

        for (size_t j = 0; j < b_query; ++j) {
            __m256i query_vec_lo = _mm256_maskload_epi32(reinterpret_cast<const int*>(query), mask_lo);
            __m256i query_vec_hi = _mm256_maskload_epi32(reinterpret_cast<const int*>(query + 4), mask_hi);
            query += num_chunks_64;

            __m256i pop_lo = popcount_avx2(_mm256_and_si256(data_vec_lo, query_vec_lo));
            __m256i pop_hi = popcount_avx2(_mm256_and_si256(data_vec_hi, query_vec_hi));
            
            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_lo);
            acc_bits[j] = _mm256_add_epi64(acc_bits[j], pop_hi);
        }
    }

    for (size_t j = 0; j < b_query; ++j) {
        __m128i shift = _mm_cvtsi32_si128(static_cast<int>(j));
        acc_ip = _mm256_add_epi64(acc_ip, _mm256_sll_epi64(acc_bits[j], shift));
    }

    // Standard reduction for a single __m256i
    auto mm256_reduce_add_epi64 = [](__m256i v) {
        __m128i low = _mm256_castsi256_si128(v);
        __m128i high = _mm256_extracti128_si256(v, 1);
        __m128i sum = _mm_add_epi64(low, high);
        return _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1);
    };

    ip_scalar += mm256_reduce_add_epi64(acc_ip);
    ppc_scalar += mm256_reduce_add_epi64(acc_ppc);

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
#else    
    std::cerr << "AVX512 VPOPCNTDQ and AVX512BW or AVX2 are required for warmup_ip_x0_q_512\n";
    exit(1);
#endif
}

template <uint32_t b_query>
inline float warmup_ip_x0_q(
    const uint64_t* data,   // pointer to data blocks (each 64 bits)
    const uint64_t* query,  // pointer to transposed query words (each 64 bits), arranged so that 
                            // for each data block the corresponding b_query query words follow
    float delta,
    float vl,
    size_t padded_dim,
    [[maybe_unused]] size_t _b_query = 0  // not used
) {
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512DQ__)
    const size_t num_blk = padded_dim / 64;
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    // Process blocks in chunks of 8
    const size_t vec_width = 8;
    size_t vec_end = (num_blk / vec_width) * vec_width;

    // Vector accumulators (each holds 8 64-bit lanes)
    __m512i ip_vec = _mm512_setzero_si512(
    );  // will accumulate weighted popcount intersections per block
    __m512i ppc_vec = _mm512_setzero_si512();  // will accumulate popcounts of data blocks

    // Loop over blocks in batches of 8
    for (size_t i = 0; i < vec_end; i += vec_width) {
        // Load eight 64-bit data blocks into x_vec.
        __m512i x_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(data + i));

        // Compute popcount for each 64-bit block in x_vec using the AVX512 VPOPCNTDQ
        // instruction. (Ensure you compile with the proper flags for VPOPCNTDQ.)
        __m512i popcnt_x_vec = _mm512_popcnt_epi64(x_vec);
        ppc_vec = _mm512_add_epi64(ppc_vec, popcnt_x_vec);

        // Process each query component (b_query is a compile-time constant, and is small).
        for (uint32_t j = 0; j < b_query; j++) {
            // the query words are transposed, thus can be loaded with a single load
            __m512i q_vec = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(query + j * num_blk + i));

            // Compute bitwise AND of data blocks and corresponding query words.
            __m512i and_vec = _mm512_and_si512(x_vec, q_vec);
            // Compute popcount on each lane.
            __m512i popcnt_and = _mm512_popcnt_epi64(and_vec);

            // Multiply by the weighting factor (1 << j) for this query position.
            __m512i weighted = _mm512_slli_epi64(popcnt_and, j);

            // Accumulate weighted popcounts for these blocks.
            ip_vec = _mm512_add_epi64(ip_vec, weighted);
        }
    }

    // Horizontally reduce the vector accumulators.
    ip_scalar  += _mm512_reduce_add_epi64(ip_vec);
    ppc_scalar += _mm512_reduce_add_epi64(ppc_vec);

    // Process remaining blocks that did not fit in the vectorized loop.
    for (size_t i = vec_end; i < num_blk; i++) {
        const uint64_t x = data[i];
        ppc_scalar += __builtin_popcountll(x);
        for (uint32_t j = 0; j < b_query; j++) {
            ip_scalar += __builtin_popcountll(x & query[j * num_blk + i]) << j;
        }
    }

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
#elif defined(__AVX2__)
    const size_t num_blk = padded_dim / 64;
    size_t ip_scalar = 0;
    size_t ppc_scalar = 0;

    // Process blocks in chunks of 4
    const size_t vec_width = 4;
    size_t vec_end = (num_blk / vec_width) * vec_width;
    
    // Accumulators
    __m256i ip_vec = _mm256_setzero_si256();
    __m256i ppc_vec = _mm256_setzero_si256();

    for (size_t i = 0; i < vec_end; i += 4) {
        // Load four 64-bit data blocks into x_vec.
        __m256i x_vec = _mm256_loadu_si256((const __m256i*)&data[i]);
        
        ppc_vec = _mm256_add_epi64(ppc_vec, popcount_avx2(x_vec));

        // Process each query component (b_query is a compile-time constant, and is small).
        for (uint32_t j = 0; j < b_query; j++) {

            // Gather query data: query[idx]
            __m256i q_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query + j * num_blk + i));

            // Compute bitwise AND of data blocks and corresponding query words.
            __m256i and_vec = _mm256_and_si256(x_vec, q_vec);
            // Compute popcount on each lane.
            __m256i popcnt_and = popcount_avx2(and_vec);

            // Multiply by the weighting factor (1 << j) for this query position.
            __m256i weighted = _mm256_slli_epi64(popcnt_and, j);
            
            // Accumulate weighted popcounts for these blocks.
            ip_vec = _mm256_add_epi64(ip_vec, weighted);
        }
    }

    auto mm256_reduce_add_epi64 = [](__m256i v) {
        __m128i low = _mm256_castsi256_si128(v);
        __m128i high = _mm256_extracti128_si256(v, 1);
        __m128i sum = _mm_add_epi64(low, high);
        return _mm_extract_epi64(sum, 0) + _mm_extract_epi64(sum, 1);
    };
    // Horizontally reduce the vector accumulators.
    ppc_scalar += mm256_reduce_add_epi64(ppc_vec);
    ip_scalar += mm256_reduce_add_epi64(ip_vec);

    // Process remaining blocks that did not fit in the vectorized loop.
    for (size_t i = vec_end; i < num_blk; i++) {
        const uint64_t x = data[i];
        ppc_scalar += __builtin_popcountll(x);
        for (uint32_t j = 0; j < b_query; j++) {
            ip_scalar += __builtin_popcountll(x & query[j * num_blk + i]) << j;
        }
    }

    return (delta * static_cast<float>(ip_scalar)) + (vl * static_cast<float>(ppc_scalar));
#else
    std::cerr << "AVX512 or AVX2 is required for warmup_ip_x0_q\n";
    exit(1);
#endif
    return 0.0f;
}

template <uint32_t b_query, uint32_t padded_dim>
inline float warmup_ip_x0_q(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t _padded_dim = 0,  // not used
    size_t _b_query = 0      // not used
) {
    auto num_blk = padded_dim / 64;
    const auto* it_data = data;
    const auto* it_query = query;

    size_t ip = 0;
    size_t ppc = 0;

    for (size_t i = 0; i < num_blk; ++i) {
        uint64_t x = *static_cast<const uint64_t*>(it_data);
        ppc += __builtin_popcountll(x);

        for (size_t j = 0; j < b_query; ++j) {
            uint64_t y = *static_cast<const uint64_t*>(it_query);
            ip += (__builtin_popcountll(x & y) << j);
            it_query++;
        }
        it_data++;
    }

    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(ppc));
}
