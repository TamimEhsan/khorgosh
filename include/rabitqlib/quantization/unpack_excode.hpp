#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

namespace rabitqlib::quant::rabitq_impl::ex_bits {

// Exact inverses of packing_rabitqplus_code, recovering one uint8 per dimension
// from the packed layout. The packers are pure bit permutations, so every
// unpacker here is a permutation too and pack->unpack round-trips exactly; the
// unit tests assert that at every width.
//
// The layouts are not uniform. Widths 3, 5 and 7 are composites: a wide plane
// plus a separate top-bit plane that is *transposed* across the 64-dim group,
// so its byte b holds bit c of dimension 8c + b. Widths 6 and 7 pack dimensions
// 48-63 two bits at a time into the spare top bits of three 16-byte blocks.
// Note also that the 4-bit plane inside the 5-bit layout pairs dimension k with
// k+16, while the standalone 4-bit layout pairs k with k+8.

namespace detail {

// Shared by widths 3, 5 and 7: one bit per dimension, transposed within the
// 64-dim group. plane[b] bit c holds the bit of dimension 8c + b.
inline void unpack_top_bit_plane(
    const uint8_t* plane, uint8_t* o_raw, size_t dim, size_t shift
) {
    for (size_t d = 0; d < dim; d += 64) {
        uint64_t bits;
        std::memcpy(&bits, plane, sizeof(bits));
        for (size_t j = 0; j < 64; ++j) {
            size_t b = j % 8;
            size_t c = j / 8;
            auto byte = static_cast<uint8_t>(bits >> (b * 8));
            o_raw[d + j] |= static_cast<uint8_t>(((byte >> c) & 1U) << shift);
        }
        plane += 8;
    }
}

// Shared by widths 6 and 7: dimensions 0-47 hold their low 6 bits in the low
// six bits of three 16-byte blocks; dimensions 48-63 have their 6 bits split
// two at a time across the top bits of the same three blocks.
inline void unpack_six_bit_body(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t k = 0; k < 16; ++k) {
            o_raw[d + k] = in[k] & 0x3FU;
            o_raw[d + 16 + k] = in[16 + k] & 0x3FU;
            o_raw[d + 32 + k] = in[32 + k] & 0x3FU;
            o_raw[d + 48 + k] = static_cast<uint8_t>(
                (in[k] >> 6) | ((in[16 + k] >> 6) << 2) | ((in[32 + k] >> 6) << 4)
            );
        }
        in += 48;
    }
}

}  // namespace detail

// ! require dim % 16 == 0
inline void unpacking_1bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    for (size_t d = 0; d < dim; d += 16) {
        uint16_t code;
        std::memcpy(&code, in, sizeof(code));
        for (size_t i = 0; i < 16; ++i) {
            o_raw[d + i] = static_cast<uint8_t>((code >> i) & 1U);
        }
        in += 2;
    }
}

// ! require dim % 64 == 0
inline void unpacking_2bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    for (size_t d = 0; d < dim; d += 64) {
        for (size_t k = 0; k < 16; ++k) {
            uint8_t c = in[k];
            o_raw[d + k] = c & 0x3U;
            o_raw[d + 16 + k] = (c >> 2) & 0x3U;
            o_raw[d + 32 + k] = (c >> 4) & 0x3U;
            o_raw[d + 48 + k] = (c >> 6) & 0x3U;
        }
        in += 16;
    }
}

// ! require dim % 64 == 0
inline void unpacking_3bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    // low two bits use the 2-bit layout, then the transposed top-bit plane
    for (size_t d = 0; d < dim; d += 64) {
        const uint8_t* two = in + ((d / 64) * 24);
        for (size_t k = 0; k < 16; ++k) {
            uint8_t c = two[k];
            o_raw[d + k] = c & 0x3U;
            o_raw[d + 16 + k] = (c >> 2) & 0x3U;
            o_raw[d + 32 + k] = (c >> 4) & 0x3U;
            o_raw[d + 48 + k] = (c >> 6) & 0x3U;
        }
    }
    for (size_t d = 0; d < dim; d += 64) {
        detail::unpack_top_bit_plane(in + ((d / 64) * 24) + 16, o_raw + d, 64, 2);
    }
}

// ! require dim % 16 == 0
inline void unpacking_4bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    for (size_t d = 0; d < dim; d += 16) {
        for (size_t k = 0; k < 8; ++k) {
            o_raw[d + k] = in[k] & 0xFU;
            o_raw[d + 8 + k] = in[k] >> 4;
        }
        in += 8;
    }
}

// ! require dim % 64 == 0
inline void unpacking_5bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    // the 4-bit plane here pairs k with k+16, unlike the standalone 4-bit layout
    for (size_t d = 0; d < dim; d += 64) {
        const uint8_t* four = in + ((d / 64) * 40);
        for (size_t k = 0; k < 16; ++k) {
            o_raw[d + k] = four[k] & 0xFU;
            o_raw[d + 16 + k] = four[k] >> 4;
            o_raw[d + 32 + k] = four[16 + k] & 0xFU;
            o_raw[d + 48 + k] = four[16 + k] >> 4;
        }
    }
    for (size_t d = 0; d < dim; d += 64) {
        detail::unpack_top_bit_plane(in + ((d / 64) * 40) + 32, o_raw + d, 64, 4);
    }
}

// ! require dim % 64 == 0
inline void unpacking_6bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    detail::unpack_six_bit_body(in, o_raw, dim);
}

// ! require dim % 64 == 0
inline void unpacking_7bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    for (size_t d = 0; d < dim; d += 64) {
        detail::unpack_six_bit_body(in + ((d / 64) * 56), o_raw + d, 64);
        detail::unpack_top_bit_plane(in + ((d / 64) * 56) + 48, o_raw + d, 64, 6);
    }
}

inline void unpacking_8bit_excode(const uint8_t* in, uint8_t* o_raw, size_t dim) {
    std::memcpy(o_raw, in, sizeof(uint8_t) * dim);
}

/**
 * @brief Inverse of packing_rabitqplus_code: recover one uint8 code per
 *        dimension from the packed representation.
 */
inline void unpacking_rabitqplus_code(
    const uint8_t* o_compact, uint8_t* o_raw, size_t dim, size_t ex_bits
) {
    if (ex_bits == 1) {
        unpacking_1bit_excode(o_compact, o_raw, dim);
    } else if (ex_bits == 2) {
        unpacking_2bit_excode(o_compact, o_raw, dim);
    } else if (ex_bits == 3) {
        unpacking_3bit_excode(o_compact, o_raw, dim);
    } else if (ex_bits == 4) {
        unpacking_4bit_excode(o_compact, o_raw, dim);
    } else if (ex_bits == 5) {
        unpacking_5bit_excode(o_compact, o_raw, dim);
    } else if (ex_bits == 6) {
        unpacking_6bit_excode(o_compact, o_raw, dim);
    } else if (ex_bits == 7) {
        unpacking_7bit_excode(o_compact, o_raw, dim);
    } else if (ex_bits == 8) {
        unpacking_8bit_excode(o_compact, o_raw, dim);
    } else {
        std::cerr << "Unsupported ex_bits in unpacking_rabitqplus_code(): " << ex_bits
                  << '\n'
                  << std::flush;
        exit(1);
    }
}

}  // namespace rabitqlib::quant::rabitq_impl::ex_bits
