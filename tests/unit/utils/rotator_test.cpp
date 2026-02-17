#include <gtest/gtest.h>
#include "rabitqlib/utils/rotator.hpp"
#include "test_helpers.hpp"
#include "test_data.hpp"
#include <vector>
#include <cmath>
#include <fstream>
#include <cstring>

using namespace rabitqlib;
using namespace rabitq_test;

class RotatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        dim = 128;
        test_data = TestDataGenerator::GenerateRandomVector(dim, -1.0f, 1.0f, 42);
    }

    void TearDown() override {
        // Clean up any temporary files
        std::remove("test_rotator.bin");
    }

    size_t dim;
    std::vector<float> test_data;
};

// Test that FhtKacRotator is chosen by default
TEST_F(RotatorTest, DefaultRotatorType) {
    Rotator<float>* rotator = choose_rotator<float>(dim);
    ASSERT_NE(rotator, nullptr);

    // FhtKacRotator pads to multiple of 64
    size_t padded_dim = rotator->size();
    EXPECT_EQ(padded_dim % 64, 0);
    EXPECT_GE(padded_dim, dim);

    delete rotator;
}

// Test rotation preserves vector norm (approximately for orthogonal rotation)
TEST_F(RotatorTest, RotationPreservesNorm) {
    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    // Calculate original norm
    float original_norm = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        original_norm += test_data[i] * test_data[i];
    }
    original_norm = std::sqrt(original_norm);

    // Rotate
    std::vector<float> rotated(padded_dim, 0.0f);
    rotator->rotate(test_data.data(), rotated.data());

    // Calculate rotated norm (considering padding with zeros)
    float rotated_norm = 0.0f;
    for (size_t i = 0; i < padded_dim; ++i) {
        rotated_norm += rotated[i] * rotated[i];
    }
    rotated_norm = std::sqrt(rotated_norm);

    // For FhtKacRotator with truncation, norm might change slightly
    // Allow some tolerance
    EXPECT_NEAR(original_norm, rotated_norm, 0.1f * original_norm);

    delete rotator;
}

// Test deterministic rotation
TEST_F(RotatorTest, DeterministicRotation) {
    // Note: FhtKacRotator uses random_device which is non-deterministic
    // For truly deterministic testing, we'd need MatrixRotator or seeded version
    // This test verifies rotation is consistent for same rotator instance

    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    std::vector<float> rotated1(padded_dim);
    std::vector<float> rotated2(padded_dim);

    rotator->rotate(test_data.data(), rotated1.data());
    rotator->rotate(test_data.data(), rotated2.data());

    // Same rotator should produce identical results
    ASSERT_VECTORS_NEARLY_EQUAL(rotated1.data(), rotated2.data(), padded_dim, 1e-6f);

    delete rotator;
}

// Test rotation of zero vector
TEST_F(RotatorTest, ZeroVectorRotation) {
    auto zero_data = TestDataGenerator::GenerateZeroVector(dim);
    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    std::vector<float> rotated(padded_dim);
    rotator->rotate(zero_data.data(), rotated.data());

    // Rotation of zero should be zero
    for (size_t i = 0; i < padded_dim; ++i) {
        EXPECT_NEAR(rotated[i], 0.0f, 1e-6f) << "Non-zero value at index " << i;
    }

    delete rotator;
}

// Test padding dimension is correct
TEST_F(RotatorTest, PaddingDimension) {
    std::vector<size_t> dims = {64, 100, 128, 256, 500, 1024};

    for (size_t d : dims) {
        Rotator<float>* rotator = choose_rotator<float>(d);
        size_t padded_dim = rotator->size();

        // FhtKacRotator pads to multiple of 64
        EXPECT_EQ(padded_dim % 64, 0) << "Padded dim not multiple of 64 for dim=" << d;
        EXPECT_GE(padded_dim, d) << "Padded dim smaller than original for dim=" << d;
        EXPECT_LT(padded_dim, d + 64) << "Excessive padding for dim=" << d;

        delete rotator;
    }
}

// Test save and load functionality with file streams
TEST_F(RotatorTest, SaveLoadFileStream) {
    Rotator<float>* rotator1 = choose_rotator<float>(dim);
    size_t padded_dim = rotator1->size();

    // Rotate with original rotator
    std::vector<float> rotated1(padded_dim);
    rotator1->rotate(test_data.data(), rotated1.data());

    // Save to file
    std::ofstream out_file("test_rotator.bin", std::ios::binary);
    ASSERT_TRUE(out_file.is_open());
    rotator1->save(out_file);
    out_file.close();

    // Create new rotator and load
    Rotator<float>* rotator2 = choose_rotator<float>(dim);
    std::ifstream in_file("test_rotator.bin", std::ios::binary);
    ASSERT_TRUE(in_file.is_open());
    rotator2->load(in_file);
    in_file.close();

    // Rotate with loaded rotator
    std::vector<float> rotated2(padded_dim);
    rotator2->rotate(test_data.data(), rotated2.data());

    // Should produce same results
    ASSERT_VECTORS_NEARLY_EQUAL(rotated1.data(), rotated2.data(), padded_dim, 1e-5f);

    delete rotator1;
    delete rotator2;
}

// Test save and load functionality with memory buffer
TEST_F(RotatorTest, SaveLoadBuffer) {
    Rotator<float>* rotator1 = choose_rotator<float>(dim);
    size_t padded_dim = rotator1->size();

    // Rotate with original rotator
    std::vector<float> rotated1(padded_dim);
    rotator1->rotate(test_data.data(), rotated1.data());

    // Save to buffer
    size_t buffer_size = rotator1->dump_bytes();
    EXPECT_GT(buffer_size, 0);

    std::vector<char> buffer(buffer_size);
    rotator1->save(buffer.data());

    // Create new rotator and load from buffer
    Rotator<float>* rotator2 = choose_rotator<float>(dim);
    rotator2->load(buffer.data());

    // Rotate with loaded rotator
    std::vector<float> rotated2(padded_dim);
    rotator2->rotate(test_data.data(), rotated2.data());

    // Should produce same results
    ASSERT_VECTORS_NEARLY_EQUAL(rotated1.data(), rotated2.data(), padded_dim, 1e-5f);

    delete rotator1;
    delete rotator2;
}

// Test rotation distributes values
TEST_F(RotatorTest, RotationDistributesValues) {
    auto simple_data = TestDataGenerator::GenerateSimpleVector(dim);
    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    std::vector<float> rotated(padded_dim);
    rotator->rotate(simple_data.data(), rotated.data());

    // After rotation, values should be distributed (not identical)
    // Count unique values (with tolerance)
    int unique_count = 0;
    std::vector<bool> is_unique(padded_dim, true);

    for (size_t i = 0; i < padded_dim - 1; ++i) {
        if (is_unique[i]) {
            unique_count++;
            for (size_t j = i + 1; j < padded_dim; ++j) {
                if (FloatNearlyEqual(rotated[i], rotated[j], 1e-3f)) {
                    is_unique[j] = false;
                }
            }
        }
    }

    // Should have good distribution
    EXPECT_GT(unique_count, static_cast<int>(padded_dim * 0.5));

    delete rotator;
}

// Test different input data types/distributions
TEST_F(RotatorTest, DifferentDistributions) {
    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t padded_dim = rotator->size();

    // Gaussian
    auto gaussian = TestDataGenerator::GenerateGaussianVector(dim, 0.0f, 1.0f, 123);
    std::vector<float> rotated_gaussian(padded_dim);
    rotator->rotate(gaussian.data(), rotated_gaussian.data());

    // Uniform
    auto uniform = TestDataGenerator::GenerateRandomVector(dim, -5.0f, 5.0f, 456);
    std::vector<float> rotated_uniform(padded_dim);
    rotator->rotate(uniform.data(), rotated_uniform.data());

    // All ones
    auto ones = TestDataGenerator::GenerateOnesVector(dim);
    std::vector<float> rotated_ones(padded_dim);
    rotator->rotate(ones.data(), rotated_ones.data());

    // All rotations should succeed without crashes
    SUCCEED();

    delete rotator;
}

// Test dump_bytes returns consistent size
TEST_F(RotatorTest, DumpBytesConsistency) {
    Rotator<float>* rotator = choose_rotator<float>(dim);
    size_t dump_size = rotator->dump_bytes();

    // Size should be positive and consistent
    EXPECT_GT(dump_size, 0);

    // Check multiple calls return same value
    EXPECT_EQ(dump_size, rotator->dump_bytes());

    delete rotator;
}
