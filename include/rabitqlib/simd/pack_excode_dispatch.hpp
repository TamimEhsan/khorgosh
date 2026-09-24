#pragma once

#include <cstddef>
#include <cstdint>

namespace rabitqlib::simd {

void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_2bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_3bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_4bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_5bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_6bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_7bit_excode_avx2(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_2bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_3bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_4bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_5bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_6bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_7bit_excode_avx512(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

// Portable (Highway) backend; compiled on every architecture. See
// docs/portability/highway-plan.md.
void packing_2bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_3bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_4bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_5bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_6bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

void packing_7bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim);

}  // namespace rabitqlib::simd
