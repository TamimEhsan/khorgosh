#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/simd/space_xy_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"

using namespace rabitqlib;

namespace {

constexpr size_t kDim = 960;

// Ground truth computed in double from the unpacked codes, independent of both
// kernels: sum(code * q_real) with q_real = delta * q_int + vl.
double ReferenceIp(
    const std::vector<uint8_t>& raw,
    const std::vector<uint8_t>& query,
    double delta,
    double vl
) {
    double acc = 0;
    for (size_t i = 0; i < raw.size(); ++i) {
        acc += static_cast<double>(raw[i]) * ((delta * query[i]) + vl);
    }
    return acc;
}

}  // namespace

TEST(XyBaseIp, MatchesReferenceAtEveryWidth) {
    std::mt19937 gen(11);
    // query width is 4 bits today; it must stay <= 7 so it fits dpbusd's signed slot
    std::uniform_int_distribution<int> qd(0, 15);

    for (size_t bits = 1; bits <= 8; ++bits) {
        std::uniform_int_distribution<int> cd(0, (1 << bits) - 1);
        std::vector<uint8_t> raw(kDim);
        std::vector<uint8_t> query(kDim);
        for (size_t i = 0; i < kDim; ++i) {
            raw[i] = static_cast<uint8_t>(cd(gen));
            query[i] = static_cast<uint8_t>(qd(gen));
        }
        std::vector<uint8_t> packed((kDim * bits / 8) + 8, 0);
        quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
            raw.data(), packed.data(), kDim, bits
        );

        const float delta = 0.0137F;
        const float vl = -0.42F;
        double want = ReferenceIp(raw, query, delta, vl);

        float got_scalar = simd::xy_base_ip_u8_scalar(
            query.data(), packed.data(), kDim, bits, delta, vl
        );
        EXPECT_NEAR(got_scalar, want, std::fabs(want) * 1e-5 + 1e-3)
            << "scalar, bits=" << bits;

        if (cpu::has_avx2()) {
            float got_avx2 = simd::xy_base_ip_u8_avx2(
                query.data(), packed.data(), kDim, bits, delta, vl
            );
            EXPECT_NEAR(got_avx2, want, std::fabs(want) * 1e-5 + 1e-3)
                << "avx2, bits=" << bits;
            EXPECT_NEAR(got_avx2, got_scalar, std::fabs(got_scalar) * 1e-6 + 1e-4)
                << "avx2 vs scalar, bits=" << bits;
        }

        if (cpu::has_avx512_vnni()) {
            float got_vnni = simd::xy_base_ip_u8_avx512vnni(
                query.data(), packed.data(), kDim, bits, delta, vl
            );
            EXPECT_NEAR(got_vnni, want, std::fabs(want) * 1e-5 + 1e-3)
                << "vnni, bits=" << bits;
            // both kernels reduce integers, so they should agree far more tightly
            EXPECT_NEAR(got_vnni, got_scalar, std::fabs(got_scalar) * 1e-6 + 1e-4)
                << "vnni vs scalar, bits=" << bits;
        }
    }
}

// base_bits == 8 pushes codes to 255, which overflows dpbusd's signed operand if
// the operands are passed in the wrong order. Saturate the codes to catch that.
TEST(XyBaseIp, MaxCodeValuesAtEightBits) {
    std::vector<uint8_t> raw(kDim, 255);
    std::vector<uint8_t> query(kDim, 15);
    std::vector<uint8_t> packed(kDim + 8, 0);
    quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
        raw.data(), packed.data(), kDim, 8
    );

    const float delta = 1.0F;
    const float vl = 0.0F;
    double want = ReferenceIp(raw, query, delta, vl);

    EXPECT_NEAR(
        simd::xy_base_ip_u8_scalar(query.data(), packed.data(), kDim, 8, delta, vl),
        want,
        1.0
    );
    if (cpu::has_avx2()) {
        // maddubs saturation shows up here first if the query ever widens
        EXPECT_NEAR(
            simd::xy_base_ip_u8_avx2(query.data(), packed.data(), kDim, 8, delta, vl),
            want,
            1.0
        );
    }
    if (cpu::has_avx512_vnni()) {
        EXPECT_NEAR(
            simd::xy_base_ip_u8_avx512vnni(
                query.data(), packed.data(), kDim, 8, delta, vl
            ),
            want,
            1.0
        );
    }
}
