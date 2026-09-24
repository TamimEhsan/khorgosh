// Portable (Highway) excode bit-packing kernels. See
// docs/portability/highway-plan.md Phase 3.
//
// Unlike new_transpose_bin[_512] (see space_highway.cpp), this packed
// layout is NOT purely transient: it is written once per vector during
// index construction and read back by excode_ipimpl::ip*_fxu* (and
// persisted as part of the index's on-disk ExDataMap-managed storage — see
// AGENTS.md's quantization/byte-layout rules), so it must match
// pack_excode_kernels.hpp's AVX2/AVX-512 byte-for-byte layout exactly, not
// merely be internally consistent. Each function below is a scalar
// re-derivation of the corresponding *_intrinsics function's bit-shift
// arithmetic (see that header for the original); this is index-construction
// code, not the search hot path, so plain scalar loops are appropriate
// (matching the reasoning already used for new_transpose_bin/warmup).

#include <cstddef>
#include <cstdint>

#include "rabitqlib/simd/pack_excode_dispatch.hpp"

namespace rabitqlib::simd {
namespace {

// Shared by the 3/5/7-bit packers below: extracts bit `shift` of each of 64
// raw codes into 8 output bytes. Re-derived from e.g.
// packing_3bit_excode_intrinsics's "top_bit" loop: that loop reads 8 raw
// bytes at a time as a little-endian uint64 `cur_codes`, computes
// `((cur_codes >> shift) & 0x0101010101010101) << (i/8)` for i = 0, 8, ...,
// 56, and ORs the results together. Shifting the whole 64-bit value by
// `shift` moves bit (8*L + shift) of byte lane L to bit 8*L (no cross-lane
// effect, since shift < 8), and the mask keeps only that bit; the outer
// `<< (i/8)` then moves byte lane L's bit from position 8*L to position
// 8*L + i/8, still within lane L since i/8 < 8. So for m = i/8 and L the
// byte lane: output bit (8*L + m) = raw byte (8*m + L)'s bit `shift`. That
// is exactly: output byte L, bit m = raw[8*m + L]'s bit `shift`.
inline void extract_bit_plane(
    const uint8_t* __restrict__ raw64, uint8_t* __restrict__ out8, unsigned shift
) {
    for (size_t lane = 0; lane < 8; ++lane) {
        uint8_t byte = 0;
        for (size_t m = 0; m < 8; ++m) {
            byte =
                static_cast<uint8_t>(byte | (((raw64[8 * m + lane] >> shift) & 1U) << m));
        }
        out8[lane] = byte;
    }
}

}  // namespace

void packing_2bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    for (size_t j = 0; j < dim; j += 64) {
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] = static_cast<uint8_t>(
                o_raw[k] | (o_raw[16 + k] << 2) | (o_raw[32 + k] << 4) |
                (o_raw[48 + k] << 6)
            );
        }
        o_raw += 64;
        o_compact += 16;
    }
}

void packing_3bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    constexpr uint8_t kMask = 0b11;
    for (size_t j = 0; j < dim; j += 64) {
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] = static_cast<uint8_t>(
                (o_raw[k] & kMask) | ((o_raw[16 + k] & kMask) << 2) |
                ((o_raw[32 + k] & kMask) << 4) | ((o_raw[48 + k] & kMask) << 6)
            );
        }
        extract_bit_plane(o_raw, o_compact + 16, 2);
        o_raw += 64;
        o_compact += 24;
    }
}

void packing_4bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 16 == 0
    for (size_t j = 0; j < dim; j += 16) {
        for (size_t k = 0; k < 8; ++k) {
            o_compact[k] = static_cast<uint8_t>(o_raw[k] | (o_raw[8 + k] << 4));
        }
        o_raw += 16;
        o_compact += 8;
    }
}

void packing_5bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    constexpr uint8_t kMask = 0b1111;
    for (size_t j = 0; j < dim; j += 64) {
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] =
                static_cast<uint8_t>((o_raw[k] & kMask) | ((o_raw[16 + k] & kMask) << 4));
            o_compact[16 + k] = static_cast<uint8_t>(
                (o_raw[32 + k] & kMask) | ((o_raw[48 + k] & kMask) << 4)
            );
        }
        extract_bit_plane(o_raw, o_compact + 32, 4);
        o_raw += 64;
        o_compact += 40;
    }
}

void packing_6bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    // for vec00 to vec47, split code into 6; for vec48 to vec63, split into
    // three 2-bit slices distributed across the three 6-bit groups.
    constexpr uint8_t kMask6 = 0b00111111;
    constexpr uint8_t kMask2 = 0b11000000;
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] =
                static_cast<uint8_t>((o_raw[k] & kMask6) | ((o_raw[48 + k] << 6) & kMask2));
            o_compact[16 + k] = static_cast<uint8_t>(
                (o_raw[16 + k] & kMask6) | ((o_raw[48 + k] << 4) & kMask2)
            );
            o_compact[32 + k] = static_cast<uint8_t>(
                (o_raw[32 + k] & kMask6) | ((o_raw[48 + k] << 2) & kMask2)
            );
        }
        o_raw += 64;
        o_compact += 48;
    }
}

void packing_7bit_excode_highway(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    // ! require dim % 64 == 0
    // Same 6-bit layout as packing_6bit_excode_highway for the low 6 bits of
    // all 64 codes, plus a 7th-bit plane (extract_bit_plane) for all 64.
    constexpr uint8_t kMask6 = 0b00111111;
    constexpr uint8_t kMask2 = 0b11000000;
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t k = 0; k < 16; ++k) {
            o_compact[k] =
                static_cast<uint8_t>((o_raw[k] & kMask6) | ((o_raw[48 + k] << 6) & kMask2));
            o_compact[16 + k] = static_cast<uint8_t>(
                (o_raw[16 + k] & kMask6) | ((o_raw[48 + k] << 4) & kMask2)
            );
            o_compact[32 + k] = static_cast<uint8_t>(
                (o_raw[32 + k] & kMask6) | ((o_raw[48 + k] << 2) & kMask2)
            );
        }
        extract_bit_plane(o_raw, o_compact + 48, 6);
        o_raw += 64;
        o_compact += 56;
    }
}

}  // namespace rabitqlib::simd
