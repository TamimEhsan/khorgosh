#include <gtest/gtest.h>
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/defines.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
#include <vector>
#include <cmath>

using namespace rabitqlib;
using namespace rabitq_test;

class SpaceTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 128;
    }

    size_t dim;
};

// Test scalar quantization with uint8_t
TEST_F(SpaceTest, ScalarQuantizeUint8) {
    auto data = TestDataGenerator::GenerateRandomVector(dim, 0.0f, 10.0f, 42);

    float lo = 0.0f;
    float hi = 10.0f;
    float delta = (hi - lo) / 255.0f;

    std::vector<uint8_t> quantized(dim);
    scalar_impl::scalar_quantize_optimized(
        quantized.data(),
        data.data(),
        dim,
        lo,
        delta
    );

    // Verify quantized values are in valid range
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_GE(quantized[i], 0);
        EXPECT_LE(quantized[i], 255);
    }

    // Verify reconstruction is approximately correct
    for (size_t i = 0; i < dim; ++i) {
        float reconstructed = lo + quantized[i] * delta;
        EXPECT_NEAR(reconstructed, data[i], delta)
            << "Reconstruction error too large at index " << i;
    }
}

// Test scalar quantization with uint16_t
TEST_F(SpaceTest, ScalarQuantizeUint16) {
    auto data = TestDataGenerator::GenerateRandomVector(dim, -5.0f, 5.0f, 123);

    float lo = -5.0f;
    float hi = 5.0f;
    float delta = (hi - lo) / 65535.0f;

    std::vector<uint16_t> quantized(dim);
    scalar_impl::scalar_quantize_optimized(
        quantized.data(),
        data.data(),
        dim,
        lo,
        delta
    );

    // Verify quantized values are in valid range
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_GE(quantized[i], 0);
        EXPECT_LE(quantized[i], 65535);
    }

    // Better precision with uint16_t
    for (size_t i = 0; i < dim; ++i) {
        float reconstructed = lo + quantized[i] * delta;
        EXPECT_NEAR(reconstructed, data[i], delta * 2.0f);
    }
}

// Test scalar quantization is deterministic
TEST_F(SpaceTest, ScalarQuantizeDeterministic) {
    auto data = TestDataGenerator::GenerateRandomVector(dim, 0.0f, 1.0f, 999);

    float lo = 0.0f;
    float delta = 1.0f / 255.0f;

    std::vector<uint8_t> quantized1(dim);
    std::vector<uint8_t> quantized2(dim);

    scalar_impl::scalar_quantize_optimized(quantized1.data(), data.data(), dim, lo, delta);
    scalar_impl::scalar_quantize_optimized(quantized2.data(), data.data(), dim, lo, delta);

    // Should produce identical results
    EXPECT_EQ(quantized1, quantized2);
}

// Test scalar quantization with zero values
TEST_F(SpaceTest, ScalarQuantizeZeros) {
    auto data = TestDataGenerator::GenerateZeroVector(dim);

    float lo = 0.0f;
    float delta = 1.0f / 255.0f;

    std::vector<uint8_t> quantized(dim);
    scalar_impl::scalar_quantize_optimized(quantized.data(), data.data(), dim, lo, delta);

    // All zeros should quantize to zero
    for (size_t i = 0; i < dim; ++i) {
        EXPECT_EQ(quantized[i], 0) << "Non-zero quantized value at index " << i;
    }
}

// Test scalar quantization with constant values
TEST_F(SpaceTest, ScalarQuantizeConstant) {
    auto data = TestDataGenerator::GenerateOnesVector(dim);

    float lo = 0.0f;
    float hi = 2.0f;
    float delta = (hi - lo) / 255.0f;

    std::vector<uint8_t> quantized(dim);
    scalar_impl::scalar_quantize_optimized(quantized.data(), data.data(), dim, lo, delta);

    // All ones should quantize to approximately the same value
    uint8_t first_val = quantized[0];
    for (size_t i = 1; i < dim; ++i) {
        EXPECT_EQ(quantized[i], first_val)
            << "Inconsistent quantization of constant values at index " << i;
    }

    // Value should be approximately (1.0 - 0.0) / delta
    uint8_t expected = static_cast<uint8_t>(std::round((1.0f - lo) / delta));
    EXPECT_EQ(first_val, expected);
}

// Test scalar quantization preserves ordering
TEST_F(SpaceTest, ScalarQuantizePreservesOrdering) {
    auto data = TestDataGenerator::GenerateIncrementalVector(dim);

    // Normalize to [0, 1] range
    for (float& val : data) {
        val = val / static_cast<float>(dim);
    }

    float lo = 0.0f;
    float delta = 1.0f / 255.0f;

    std::vector<uint8_t> quantized(dim);
    scalar_impl::scalar_quantize_optimized(quantized.data(), data.data(), dim, lo, delta);

    // Quantized values should be non-decreasing (monotonic)
    for (size_t i = 1; i < dim; ++i) {
        EXPECT_GE(quantized[i], quantized[i-1])
            << "Ordering not preserved at index " << i;
    }
}

// Test MetricType enum
TEST_F(SpaceTest, MetricTypeEnum) {
    // Just verify the enum values exist
    MetricType l2 = METRIC_L2;
    MetricType ip = METRIC_IP;

    EXPECT_EQ(l2, METRIC_L2);
    EXPECT_EQ(ip, METRIC_IP);
    EXPECT_NE(l2, ip);
}

// Test AnnCandidate struct
TEST_F(SpaceTest, AnnCandidateOrdering) {
    AnnCandidate<float> c1(1, 0.5f);
    AnnCandidate<float> c2(2, 0.3f);
    AnnCandidate<float> c3(3, 0.7f);

    // Test comparison operators
    EXPECT_LT(c2, c1);  // 0.3 < 0.5
    EXPECT_LT(c1, c3);  // 0.5 < 0.7
    EXPECT_GT(c3, c2);  // 0.7 > 0.3

    EXPECT_LE(c2, c1);
    EXPECT_GE(c3, c1);
}

// Test AnnCandidate default initialization
TEST_F(SpaceTest, AnnCandidateDefaultInit) {
    AnnCandidate<float> candidate;

    EXPECT_EQ(candidate.id, 0u);
    EXPECT_EQ(candidate.distance, std::numeric_limits<float>::max());
}

// Test scalar quantization with different ranges
TEST_F(SpaceTest, ScalarQuantizeDifferentRanges) {
    std::vector<std::pair<float, float>> ranges = {
        {0.0f, 1.0f},
        {-1.0f, 1.0f},
        {-10.0f, 10.0f},
        {0.0f, 100.0f}
    };

    for (const auto& [lo, hi] : ranges) {
        auto data = TestDataGenerator::GenerateRandomVector(dim, lo, hi, 555);
        float delta = (hi - lo) / 255.0f;

        std::vector<uint8_t> quantized(dim);
        scalar_impl::scalar_quantize_optimized(quantized.data(), data.data(), dim, lo, delta);

        // Verify reconstruction quality
        float max_error = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            float reconstructed = lo + quantized[i] * delta;
            float error = std::abs(reconstructed - data[i]);
            max_error = std::max(max_error, error);
        }

        EXPECT_LT(max_error, delta * 1.5f)
            << "Max error too large for range [" << lo << ", " << hi << "]";
    }
}
