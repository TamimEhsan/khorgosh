// Portable (Highway) FastScan accumulation kernels. See
// docs/portability/highway-plan.md Phase 3.
//
// pack_codes (fastscan.hpp) rearranges each 32-vector batch's byte-column i
// (holding codebooks 2i and 2i+1, one nibble each) into a 32-byte block via
// kPerm0: for j in [0, 16), letting a = original byte i of row kPerm0[j]
// and b = original byte i of row kPerm0[j] + 16,
//   block[j]      = (a >> 4) | (b & 0xF0)   // high nibbles of a, b
//   block[j + 16] = (a & 0x0F) | (b << 4)   // low nibbles of a, b
// (block[j] holds codebook 2i's code since group 2i is even -> high nibble;
// block[j + 16] holds codebook 2i+1's code, the low nibble). This directly
// inverts that layout by iterating j and writing both recovered rows
// (kPerm0[j] and kPerm0[j] + 16) at once, without needing kPerm0's inverse.
// Table lookups indexed by extracted nibbles are a gather, which has no
// portable vectorized form across Highway targets, so — like
// pack_excode_highway.cpp and warmup_highway.cpp — this stays plain scalar;
// only the int64_t accumulator (matching the AVX2 backend's overflow
// semantics without needing its chunked-int16 workaround) differs from a
// straight port.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"

namespace rabitqlib::fastscan::simd {
namespace {

template <class LookupFn>
void accumulate_generic(
    const uint8_t* __restrict__ codes, size_t dim, int64_t* totals, LookupFn lookup
) {
    const size_t cols = dim / 8;
    for (size_t i = 0; i < cols; ++i) {
        const uint8_t* block = codes + (i * 32);
        const size_t group_even = 2 * i;
        const size_t group_odd = (2 * i) + 1;
        for (size_t j = 0; j < 16; ++j) {
            const uint8_t byte_j = block[j];
            const uint8_t byte_j16 = block[j + 16];
            totals[kPerm0[j]] +=
                lookup(group_even, byte_j & 0xFU) + lookup(group_odd, byte_j16 & 0xFU);
            totals[kPerm0[j] + 16] += lookup(group_even, (byte_j >> 4) & 0xFU) +
                                      lookup(group_odd, (byte_j16 >> 4) & 0xFU);
        }
    }
}

void check_no_overflow(const int64_t* totals, const char* message) {
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        if (totals[lane] > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error(message);
        }
    }
}

}  // namespace

void accumulate_highway(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    int32_t* __restrict__ result,
    size_t dim
) {
    int64_t totals[kBatchSize] = {};
    accumulate_generic(codes, dim, totals, [&](size_t group, uint8_t code) -> int64_t {
        return lp_table[(group * 16) + code];
    });
    check_no_overflow(totals, "FastScan result exceeds int32_t");
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        result[lane] = static_cast<int32_t>(totals[lane]);
    }
}

// hc_lut's layout is transient, per-query data produced by transfer_lut_hacc
// and immediately consumed by accumulate_hacc within the same dispatch
// backend (see AGENTS.md: only shared/persisted layouts must match the
// AVX2/AVX-512 byte-for-byte layout) — so this uses a simple interleaved
// low-byte/high-byte-per-entry layout rather than replicating
// transfer_lut_hacc_avx2's SIMD-shuffle-friendly arrangement.
void transfer_lut_hacc_highway(
    const uint16_t* __restrict__ lut, size_t dim, uint8_t* __restrict__ hc_lut
) {
    const size_t num_entries = dim * 4;
    for (size_t e = 0; e < num_entries; ++e) {
        hc_lut[2 * e] = static_cast<uint8_t>(lut[e]);
        hc_lut[(2 * e) + 1] = static_cast<uint8_t>(lut[e] >> 8);
    }
}

void accumulate_hacc_highway(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
    int64_t totals[kBatchSize] = {};
    accumulate_generic(codes, dim, totals, [&](size_t group, uint8_t code) -> int64_t {
        const size_t idx = ((group * 16) + code) * 2;
        return static_cast<uint16_t>(
            static_cast<uint16_t>(hc_lut[idx]) |
            (static_cast<uint16_t>(hc_lut[idx + 1]) << 8)
        );
    });
    check_no_overflow(totals, "high-accuracy FastScan result exceeds int32_t");
    for (size_t lane = 0; lane < kBatchSize; ++lane) {
        accu_res[lane] = static_cast<int32_t>(totals[lane]);
    }
}

}  // namespace rabitqlib::fastscan::simd
