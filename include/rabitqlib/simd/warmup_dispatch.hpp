#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

float warmup_ip_x0_q_512_avx2(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

float warmup_ip_x0_q_512_avx2(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

float warmup_ip_x0_q_512_avx512(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

float warmup_ip_x0_q_512_avx512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

// Portable (Highway) backend; compiled on every architecture. See
// docs/portability/highway-plan.md.
float warmup_ip_x0_q_512_highway(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

float warmup_ip_x0_q_512_highway(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
);

// HNSW-internal fast path for the fixed 4-bit case
// (SplitSingleQuery<float>::kNumBits); not a general-purpose entry point —
// see its definition in warmup_highway.cpp. Only
// HnswHighwayKernel::warmup_ip_x0_q_512 (dispatch_highway.cpp) calls this.
float warmup_ip_x0_q_512_bits4_highway(
    const uint8_t* data, const uint64_t* query, float delta, float vl, size_t padded_dim
);

}  // namespace rabitqlib::simd
