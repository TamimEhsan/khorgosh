#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

// Inner product between a scalar-quantized query (one uint8 per dimension) and a
// packed base code, returning the real-valued ip. The query was quantized as
// q_real ~= delta * q_int + vl, so
//     sum(code * q_real) = delta * sum(code * q_int) + vl * sum(code)
// and the kernel accumulates both terms in one pass over the unpacked codes.
float xy_base_ip_u8_avx512vnni(
    const uint8_t* quant_query,
    const uint8_t* code,
    size_t dim,
    size_t base_bits,
    float delta,
    float vl
);

float xy_base_ip_u8_avx2(
    const uint8_t* quant_query,
    const uint8_t* code,
    size_t dim,
    size_t base_bits,
    float delta,
    float vl
);

float xy_base_ip_u8_scalar(
    const uint8_t* quant_query,
    const uint8_t* code,
    size_t dim,
    size_t base_bits,
    float delta,
    float vl
);

}  // namespace rabitqlib::simd
