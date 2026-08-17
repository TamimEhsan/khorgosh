#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/quantization/rabitq_impl.hpp"
#include "rabitqlib/utils/space.hpp"
#include "test_helpers.hpp"

using namespace rabitqlib;
using namespace rabitq_test;

namespace {

std::vector<float> RandomVec(size_t dim, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> v(dim);
    for (auto& x : v) {
        x = dist(gen);
    }
    return v;
}

}  // namespace

// Backward-compat anchor: at base_bits == 1, xy_split_code_with_factor must
// match today's one_bit_code + ex_bits_code_with_factor bit-for-bit.
TEST(XyQuantization, BaseBitsOneMatchesExistingOneBitPlusExBits) {
    constexpr size_t kDim = 64;
    constexpr size_t kExtraBits = 3;

    std::mt19937 gen(42);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid(kDim, 0.0F);

    // Reference: today's separate one_bit_code (sign) + ex_bits_code_with_factor.
    std::vector<int> ref_binary_code(kDim);
    quant::rabitq_impl::one_bit::one_bit_code(
        data.data(), centroid.data(), kDim, ref_binary_code.data()
    );

    std::vector<uint8_t> ref_ex_code(kDim);
    float ref_f_add = 0;
    float ref_f_rescale = 0;
    float ref_f_error = 0;
    quant::rabitq_impl::ex_bits::ex_bits_code_with_factor<float, uint8_t>(
        data.data(),
        centroid.data(),
        kDim,
        kExtraBits,
        ref_ex_code.data(),
        ref_f_add,
        ref_f_rescale,
        ref_f_error,
        METRIC_L2
    );

    // New path: base_bits=1, extra_bits=kExtraBits.
    std::vector<uint8_t> base_code(kDim);
    std::vector<uint8_t> extra_code(kDim);
    float f_add = 0;
    float f_rescale = 0;
    float f_error = 0;
    quant::rabitq_impl::xy_bits::xy_split_code_with_factor<float, uint8_t>(
        data.data(),
        centroid.data(),
        kDim,
        /*base_bits=*/1,
        /*extra_bits=*/kExtraBits,
        base_code.data(),
        extra_code.data(),
        f_add,
        f_rescale,
        f_error,
        METRIC_L2
    );

    for (size_t i = 0; i < kDim; ++i) {
        EXPECT_EQ(base_code[i], static_cast<uint8_t>(ref_binary_code[i])) << "dim " << i;
        EXPECT_EQ(extra_code[i], ref_ex_code[i]) << "dim " << i;
    }
    EXPECT_FLOAT_NEARLY_EQUAL(f_add, ref_f_add, 1e-4F);
    EXPECT_FLOAT_NEARLY_EQUAL(f_rescale, ref_f_rescale, 1e-4F);
    EXPECT_FLOAT_NEARLY_EQUAL(f_error, ref_f_error, 1e-4F);
}

// ip(base)*2^extra_bits + ip(extra) must equal ip(total_code) -- the
// linearity xy_distance() relies on.
TEST(XyQuantization, SplitInnerProductRecombinesExactly) {
    constexpr size_t kDim = 64;
    constexpr size_t kBaseBits = 3;
    constexpr size_t kExtraBits = 4;

    std::mt19937 gen(7);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid(kDim, 0.0F);
    std::vector<float> query = RandomVec(kDim, gen);

    std::vector<uint8_t> base_code(kDim);
    std::vector<uint8_t> extra_code(kDim);
    float f_add = 0;
    float f_rescale = 0;
    float f_error = 0;
    quant::rabitq_impl::xy_bits::xy_split_code_with_factor<float, uint8_t>(
        data.data(),
        centroid.data(),
        kDim,
        kBaseBits,
        kExtraBits,
        base_code.data(),
        extra_code.data(),
        f_add,
        f_rescale,
        f_error,
        METRIC_L2
    );

    double split_ip = 0;
    double direct_ip = 0;
    for (size_t i = 0; i < kDim; ++i) {
        uint32_t total = (static_cast<uint32_t>(base_code[i]) << kExtraBits) |
                          static_cast<uint32_t>(extra_code[i]);
        direct_ip += static_cast<double>(query[i]) * static_cast<double>(total);
    }
    for (size_t i = 0; i < kDim; ++i) {
        split_ip += static_cast<double>(query[i]) * static_cast<double>(base_code[i]) *
                     static_cast<double>(1U << kExtraBits);
        split_ip += static_cast<double>(query[i]) * static_cast<double>(extra_code[i]);
    }

    EXPECT_NEAR(split_ip, direct_ip, 1e-6 * std::abs(direct_ip) + 1e-6);
}

// End-to-end backward-compat check: at base_bits=1, quantize_xy_single +
// xy_distance must agree with today's production formula
// ((1<<ex_bits)*ip_x0_qr + ip_func_(ex_code) + kbxsumq), using a plain
// reference dot product in place of the popcount kernel for ip_x0_qr.
TEST(XyQuantization, EndToEndMatchesExistingFormulaAtBaseBitsOne) {
    constexpr size_t kDim = 128;
    constexpr size_t kExtraBits = 5;

    std::mt19937 gen(123);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid = RandomVec(kDim, gen);
    std::vector<float> query = RandomVec(kDim, gen);

    constexpr float kGAdd = 3.5F;
    constexpr float kGError = 0.75F;

    // --- Reference: today's production formula, built from the existing,
    // unchanged one_bit_code + ex_bits_code_with_factor functions. ---
    std::vector<int> ref_binary_code(kDim);
    quant::rabitq_impl::one_bit::one_bit_code(
        data.data(), centroid.data(), kDim, ref_binary_code.data()
    );

    std::vector<uint8_t> ref_ex_code(kDim);
    float ref_f_add_ex = 0;
    float ref_f_rescale_ex = 0;
    float ref_f_error_ex = 0;
    quant::rabitq_impl::ex_bits::ex_bits_code_with_factor<float, uint8_t>(
        data.data(),
        centroid.data(),
        kDim,
        kExtraBits,
        ref_ex_code.data(),
        ref_f_add_ex,
        ref_f_rescale_ex,
        ref_f_error_ex,
        METRIC_L2
    );

    double ip_x0_qr = 0;
    double ex_ip = 0;
    double sumq = 0;
    for (size_t i = 0; i < kDim; ++i) {
        ip_x0_qr += static_cast<double>(query[i]) * static_cast<double>(ref_binary_code[i]);
        ex_ip += static_cast<double>(query[i]) * static_cast<double>(ref_ex_code[i]);
        sumq += query[i];
    }
    double ref_c_b = -(static_cast<double>(1 << (kExtraBits + 1)) - 1) / 2.0;
    double ref_kbxsumq = sumq * ref_c_b;

    float ref_est = static_cast<float>(
        ref_f_add_ex + kGAdd +
        (ref_f_rescale_ex *
         (static_cast<double>(1 << kExtraBits) * ip_x0_qr + ex_ip + ref_kbxsumq))
    );

    // --- New path ---
    std::vector<char> xy_data(XYDataMap<float>::data_bytes(kDim, 1, kExtraBits));
    quant::quantize_xy_single(
        data.data(), centroid.data(), kDim, /*base_bits=*/1, kExtraBits, xy_data.data()
    );

    XYQuery<float> q_obj(query.data(), kDim, /*base_bits=*/1, kExtraBits, METRIC_L2);

    auto base_ip_func = select_excode_ipfunc(1);
    auto extra_ip_func = select_excode_ipfunc(kExtraBits);

    float new_est = 0;
    float new_low = 0;
    xy_distance(
        xy_data.data(),
        base_ip_func,
        extra_ip_func,
        q_obj,
        kDim,
        /*base_bits=*/1,
        kExtraBits,
        new_est,
        new_low,
        kGAdd,
        kGError
    );

    EXPECT_FLOAT_NEARLY_EQUAL(new_est, ref_est, 1e-2F);
    EXPECT_FLOAT_NEARLY_EQUAL(new_low, new_est - (ref_f_error_ex * kGError), 1e-4F);
}

TEST(XyQuantization, CombinedBitsBeyondCapAborts) {
    constexpr size_t kDim = 64;
    std::mt19937 gen(1);
    std::vector<float> data = RandomVec(kDim, gen);
    std::vector<float> centroid(kDim, 0.0F);

    std::vector<uint8_t> base_code(kDim);
    std::vector<uint8_t> extra_code(kDim);
    float f_add = 0;
    float f_rescale = 0;
    float f_error = 0;

    auto call_with_bad_bits = [&]() {
        quant::rabitq_impl::xy_bits::xy_split_code_with_factor<float, uint8_t>(
            data.data(),
            centroid.data(),
            kDim,
            /*base_bits=*/8,
            /*extra_bits=*/8,
            base_code.data(),
            extra_code.data(),
            f_add,
            f_rescale,
            f_error,
            METRIC_L2
        );
    };
    EXPECT_DEATH(call_with_bad_bits(), "");
}
