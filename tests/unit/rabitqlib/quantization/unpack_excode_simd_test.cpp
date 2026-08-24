// The register-level unpackers live in headers under src/simd because they must
// inline into the search kernel, so this file is compiled with AVX flags of its
// own (see tests/CMakeLists.txt). The flags are per-source, not per-target: the
// other test files stay flag-free and the binary still loads on machines
// without AVX-512, where the guarded assertions simply do not run.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/quantization/unpack_excode.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "simd/unpack_excode_avx2.hpp"
#include "simd/unpack_excode_avx512.hpp"

using namespace rabitqlib;

namespace {

// The scalar unpackers are the oracle: the round-trip tests in
// unpack_excode_test.cpp pin them against the packers, so any divergence here
// is the SIMD side's fault.
void ExpectSimdMatchesScalar(size_t bits, const std::vector<uint8_t>& packed) {
    alignas(64) uint8_t want[64];
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(packed.data(), want, 64, bits);

    if (cpu::has_avx512_core()) {
        alignas(64) uint8_t got[64];
        __m512i v;
        simd::unpack64_excode_avx512(packed.data(), bits, v);
        _mm512_storeu_si512(reinterpret_cast<__m512i*>(got), v);
        for (size_t i = 0; i < 64; ++i) {
            ASSERT_EQ(got[i], want[i]) << "avx512 bits=" << bits << " dim " << i;
        }
    }

    if (cpu::has_avx2()) {
        alignas(32) uint8_t got[64];
        __m256i lo;
        __m256i hi;
        simd::unpack64_excode_avx2(packed.data(), bits, lo, hi);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(got), lo);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(got + 32), hi);
        for (size_t i = 0; i < 64; ++i) {
            ASSERT_EQ(got[i], want[i]) << "avx2 bits=" << bits << " dim " << i;
        }
    }
}

std::vector<uint8_t> PackRandom(size_t bits, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> cd(0, (1 << bits) - 1);
    std::vector<uint8_t> raw(64);
    for (auto& x : raw) {
        x = static_cast<uint8_t>(cd(gen));
    }
    std::vector<uint8_t> packed((64 * bits / 8) + 8, 0);
    quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
        raw.data(), packed.data(), 64, bits
    );
    return packed;
}

}  // namespace

TEST(UnpackExcodeSimd, MatchesScalarAtEveryWidth) {
    for (size_t bits = 1; bits <= 8; ++bits) {
        ExpectSimdMatchesScalar(bits, PackRandom(bits, 4242 + bits));
    }
}

// All-ones exercises the top of every field, which is where a mis-sized mask or
// a shift in the wrong direction shows up.
TEST(UnpackExcodeSimd, MatchesScalarAtSaturatedCodes) {
    for (size_t bits = 1; bits <= 8; ++bits) {
        std::vector<uint8_t> raw(64, static_cast<uint8_t>((1 << bits) - 1));
        std::vector<uint8_t> packed((64 * bits / 8) + 8, 0);
        quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
            raw.data(), packed.data(), 64, bits
        );
        ExpectSimdMatchesScalar(bits, packed);
    }
}
