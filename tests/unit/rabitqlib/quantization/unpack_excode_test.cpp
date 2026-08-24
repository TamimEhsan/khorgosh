#include <gtest/gtest.h>

#include <cstdint>
#include <cmath>
#include <random>
#include <vector>

#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/quantization/unpack_excode.hpp"
#include "rabitqlib/utils/space.hpp"

using namespace rabitqlib::quant::rabitq_impl::ex_bits;

namespace {

// dim % 64 == 0 satisfies every width's alignment requirement at once.
constexpr size_t kDim = 960;

std::vector<uint8_t> RandomCodes(size_t dim, size_t bits, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> dist(0, (1 << bits) - 1);
    std::vector<uint8_t> v(dim);
    for (auto& x : v) {
        x = static_cast<uint8_t>(dist(gen));
    }
    return v;
}

// pack -> unpack must be the identity for every width. The packers are pure bit
// permutations, so anything less than exact equality means an unpacker is wrong.
void ExpectRoundTrip(size_t bits) {
    std::vector<uint8_t> raw = RandomCodes(kDim, bits, 1000 + bits);
    std::vector<uint8_t> packed((kDim * bits / 8) + 8, 0);
    std::vector<uint8_t> out(kDim, 0);

    packing_rabitqplus_code(raw.data(), packed.data(), kDim, bits);
    unpacking_rabitqplus_code(packed.data(), out.data(), kDim, bits);

    for (size_t i = 0; i < kDim; ++i) {
        ASSERT_EQ(out[i], raw[i]) << "bits=" << bits << " dim " << i;
    }
}

}  // namespace

TEST(UnpackExcode, RoundTrip1Bit) { ExpectRoundTrip(1); }
TEST(UnpackExcode, RoundTrip2Bit) { ExpectRoundTrip(2); }
TEST(UnpackExcode, RoundTrip3Bit) { ExpectRoundTrip(3); }
TEST(UnpackExcode, RoundTrip4Bit) { ExpectRoundTrip(4); }
TEST(UnpackExcode, RoundTrip5Bit) { ExpectRoundTrip(5); }
TEST(UnpackExcode, RoundTrip6Bit) { ExpectRoundTrip(6); }
TEST(UnpackExcode, RoundTrip7Bit) { ExpectRoundTrip(7); }
TEST(UnpackExcode, RoundTrip8Bit) { ExpectRoundTrip(8); }

// Every value of every width must survive, not just the random draw above.
TEST(UnpackExcode, RoundTripAllValuesPerWidth) {
    for (size_t bits = 1; bits <= 8; ++bits) {
        const int max_val = (1 << bits) - 1;
        std::vector<uint8_t> raw(kDim);
        for (size_t i = 0; i < kDim; ++i) {
            raw[i] = static_cast<uint8_t>(i % (max_val + 1));
        }
        std::vector<uint8_t> packed((kDim * bits / 8) + 8, 0);
        std::vector<uint8_t> out(kDim, 0);

        packing_rabitqplus_code(raw.data(), packed.data(), kDim, bits);
        unpacking_rabitqplus_code(packed.data(), out.data(), kDim, bits);

        for (size_t i = 0; i < kDim; ++i) {
            ASSERT_EQ(out[i], raw[i]) << "bits=" << bits << " dim " << i;
        }
    }
}

// The unpacked codes must agree with what the search path actually reads: a
// one-hot query through the ex-code IP kernel returns code[j] for dimension j.
TEST(UnpackExcode, AgreesWithExcodeIpKernel) {
    for (size_t bits = 1; bits <= 8; ++bits) {
        std::vector<uint8_t> raw = RandomCodes(kDim, bits, 7 + bits);
        std::vector<uint8_t> packed((kDim * bits / 8) + 8, 0);
        std::vector<uint8_t> out(kDim, 0);
        packing_rabitqplus_code(raw.data(), packed.data(), kDim, bits);
        unpacking_rabitqplus_code(packed.data(), out.data(), kDim, bits);

        auto ip = rabitqlib::select_excode_ipfunc(bits);
        std::vector<float> probe(kDim, 0.0F);
        for (size_t j = 0; j < kDim; j += 97) {  // sample, one kernel call per probe
            probe[j] = 1.0F;
            auto seen = static_cast<uint8_t>(std::lround(ip(probe.data(), packed.data(), kDim)));
            probe[j] = 0.0F;
            ASSERT_EQ(out[j], seen) << "bits=" << bits << " dim " << j;
        }
    }
}
