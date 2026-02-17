#include <gtest/gtest.h>
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
#include <vector>
#include <cmath>

using namespace rabitqlib;
using namespace rabitq_test;

class RabitQTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common dimensions and bit widths for testing
        dim = 128;
        bit_widths = {1, 2, 4, 8};
    }

    size_t dim;
    std::vector<size_t> bit_widths;
};

// Test basic quantization and reconstruction with scalar quantizer
TEST_F(RabitQTest, ScalarQuantizeAndReconstruct) {
    for (size_t bit : bit_widths) {
        auto data = TestDataGenerator::GenerateRandomVector(dim, -1.0f, 1.0f, 42);

        // Choose rotator and rotate data
        Rotator<float>* rotator = choose_rotator<float>(dim);
        size_t padded_dim = rotator->size();

        std::vector<float> rotated_data(padded_dim);
        rotator->rotate(data.data(), rotated_data.data());

        // Quantize
        std::vector<uint8_t> code(padded_dim);
        float delta = 0.0f;
        float vl = 0.0f;

        quant::quantize_scalar(
            rotated_data.data(),
            padded_dim,
            bit,
            code.data(),
            delta,
            vl
        );

        // Reconstruct
        std::vector<float> reconstructed(padded_dim);
        quant::reconstruct_vec(code.data(), delta, vl, padded_dim, reconstructed.data());

        // Calculate reconstruction error
        float mse = MeanSquaredError(rotated_data.data(), reconstructed.data(), padded_dim);

        // Higher bit widths should give lower error
        // For 4-bit, MSE should be reasonably low
        if (bit >= 4) {
            EXPECT_LT(mse, 0.01f) << "MSE too high for " << bit << " bits";
        }

        delete rotator;
    }
}

// Test that quantization is deterministic
TEST_F(RabitQTest, DeterministicQuantization) {
    size_t bit = 4;
    auto data = TestDataGenerator::GenerateRandomVector(dim, -1.0f, 1.0f, 123);

    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    std::vector<float> rotated_data(padded_dim);
    rotator->rotate(data.data(), rotated_data.data());

    // Quantize twice
    std::vector<uint8_t> code1(padded_dim);
    float delta1 = 0.0f, vl1 = 0.0f;
    quant::quantize_scalar(rotated_data.data(), padded_dim, bit, code1.data(), delta1, vl1);

    std::vector<uint8_t> code2(padded_dim);
    float delta2 = 0.0f, vl2 = 0.0f;
    quant::quantize_scalar(rotated_data.data(), padded_dim, bit, code2.data(), delta2, vl2);

    // Results should be identical
    EXPECT_FLOAT_EQ(delta1, delta2);
    EXPECT_FLOAT_EQ(vl1, vl2);
    EXPECT_EQ(code1, code2);

    delete rotator;
}

// Test zero vector quantization
TEST_F(RabitQTest, ZeroVectorQuantization) {
    size_t bit = 4;
    auto data = TestDataGenerator::GenerateZeroVector(dim);

    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    std::vector<float> rotated_data(padded_dim);
    rotator->rotate(data.data(), rotated_data.data());

    std::vector<uint8_t> code(padded_dim);
    float delta = 0.0f, vl = 0.0f;
    quant::quantize_scalar(rotated_data.data(), padded_dim, bit, code.data(), delta, vl);

    std::vector<float> reconstructed(padded_dim);
    quant::reconstruct_vec(code.data(), delta, vl, padded_dim, reconstructed.data());

    // Reconstructed should be close to zero
    for (size_t i = 0; i < padded_dim; ++i) {
        EXPECT_NEAR(reconstructed[i], 0.0f, 1e-4f) << "Non-zero value at index " << i;
    }

    delete rotator;
}

// Test that higher bit widths give better reconstruction
TEST_F(RabitQTest, HigherBitsBetterReconstruction) {
    auto data = TestDataGenerator::GenerateRandomVector(dim, -1.0f, 1.0f, 999);

    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    std::vector<float> rotated_data(padded_dim);
    rotator->rotate(data.data(), rotated_data.data());

    std::vector<float> errors;

    for (size_t bit : {2, 4, 8}) {
        std::vector<uint8_t> code(padded_dim);
        float delta = 0.0f, vl = 0.0f;
        quant::quantize_scalar(rotated_data.data(), padded_dim, bit, code.data(), delta, vl);

        std::vector<float> reconstructed(padded_dim);
        quant::reconstruct_vec(code.data(), delta, vl, padded_dim, reconstructed.data());

        float mse = MeanSquaredError(rotated_data.data(), reconstructed.data(), padded_dim);
        errors.push_back(mse);
    }

    // Each higher bit width should reduce error (or at least not increase it significantly)
    EXPECT_LT(errors[1], errors[0] * 1.1f) << "4-bit should be better than 2-bit";
    EXPECT_LT(errors[2], errors[1] * 1.1f) << "8-bit should be better than 4-bit";

    delete rotator;
}

// Test quantization with different data distributions
TEST_F(RabitQTest, DifferentDistributions) {
    size_t bit = 4;
    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    // Test with Gaussian distribution
    auto gaussian_data = TestDataGenerator::GenerateGaussianVector(dim, 0.0f, 1.0f, 555);
    std::vector<float> rotated_gaussian(padded_dim);
    rotator->rotate(gaussian_data.data(), rotated_gaussian.data());

    std::vector<uint8_t> code(padded_dim);
    float delta = 0.0f, vl = 0.0f;
    quant::quantize_scalar(rotated_gaussian.data(), padded_dim, bit, code.data(), delta, vl);

    std::vector<float> reconstructed(padded_dim);
    quant::reconstruct_vec(code.data(), delta, vl, padded_dim, reconstructed.data());

    float mse = MeanSquaredError(rotated_gaussian.data(), reconstructed.data(), padded_dim);
    EXPECT_LT(mse, 0.1f) << "Gaussian distribution quantization error too high";

    // Test with uniform distribution
    auto uniform_data = TestDataGenerator::GenerateRandomVector(dim, 0.0f, 10.0f, 666);
    std::vector<float> rotated_uniform(padded_dim);
    rotator->rotate(uniform_data.data(), rotated_uniform.data());

    quant::quantize_scalar(rotated_uniform.data(), padded_dim, bit, code.data(), delta, vl);
    quant::reconstruct_vec(code.data(), delta, vl, padded_dim, reconstructed.data());

    mse = MeanSquaredError(rotated_uniform.data(), reconstructed.data(), padded_dim);
    EXPECT_LT(mse, 0.5f) << "Uniform distribution quantization error too high";

    delete rotator;
}

// Test delta and vl relationship
TEST_F(RabitQTest, DeltaVlRelationship) {
    size_t bit = 4;
    auto data = TestDataGenerator::GenerateRandomVector(dim, -2.0f, 2.0f, 777);

    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    std::vector<float> rotated_data(padded_dim);
    rotator->rotate(data.data(), rotated_data.data());

    std::vector<uint8_t> code(padded_dim);
    float delta = 0.0f, vl = 0.0f;
    quant::quantize_scalar(rotated_data.data(), padded_dim, bit, code.data(), delta, vl);

    // Delta should be positive for non-zero data
    EXPECT_GT(delta, 0.0f) << "Delta should be positive";

    // Verify the relationship: vl = -delta * (2^bit - 1) / 2 (as mentioned in sample code)
    float expected_vl = -delta * (std::pow(2, bit) - 1) / 2.0f;
    EXPECT_FLOAT_NEARLY_EQUAL(vl, expected_vl, 1e-4f);

    delete rotator;
}

// Test reconstruction with edge values
TEST_F(RabitQTest, EdgeValuesReconstruction) {
    size_t bit = 4;
    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    // Test with all ones
    auto ones_data = TestDataGenerator::GenerateOnesVector(dim);
    std::vector<float> rotated_ones(padded_dim);
    rotator->rotate(ones_data.data(), rotated_ones.data());

    std::vector<uint8_t> code(padded_dim);
    float delta = 0.0f, vl = 0.0f;
    quant::quantize_scalar(rotated_ones.data(), padded_dim, bit, code.data(), delta, vl);

    std::vector<float> reconstructed(padded_dim);
    quant::reconstruct_vec(code.data(), delta, vl, padded_dim, reconstructed.data());

    // Should reconstruct reasonably well
    float mse = MeanSquaredError(rotated_ones.data(), reconstructed.data(), padded_dim);
    EXPECT_LT(mse, 0.01f);

    delete rotator;
}
