#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <utility>
#include <vector>

#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/space.hpp"

namespace {
using rabitqlib::dot_product;
using rabitqlib::euclidean_sqr;

std::vector<float> RandomMat(size_t n, size_t dim, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.F, 1.F);
    std::vector<float> out(n * dim);
    for (auto& v : out) {
        v = dist(gen);
    }
    return out;
}

// The kernels block rows 8 at a time and vectorise 16 (AVX-512) or 8 (AVX2)
// lanes at a time, so the interesting sizes are the ones that do not divide
// evenly: n exercises the scalar row remainder, dim the masked lane tail.
const std::vector<size_t> kRowCounts = {1, 7, 8, 9, 16, 23, 64};
const std::vector<size_t> kDims = {1, 3, 8, 15, 16, 17, 63, 64, 128, 960};

using L2Fn = void (*)(const float*, const float*, size_t, size_t, float*);
using DotL2Fn = void (*)(const float*, const float*, size_t, size_t, float*, float*);

// Dispatch only ever reaches one of these per machine, so name them explicitly:
// otherwise the AVX2 kernel goes untested anywhere with AVX-512.
std::vector<std::pair<const char*, L2Fn>> L2Kernels() {
    std::vector<std::pair<const char*, L2Fn>> out;
    if (rabitqlib::cpu::has_avx2()) {
        out.emplace_back("avx2", rabitqlib::simd::l2_sqr_batch_avx2);
    }
    if (rabitqlib::cpu::has_avx512_core()) {
        out.emplace_back("avx512", rabitqlib::simd::l2_sqr_batch_avx512);
    }
    return out;
}

std::vector<std::pair<const char*, DotL2Fn>> DotL2Kernels() {
    std::vector<std::pair<const char*, DotL2Fn>> out;
    if (rabitqlib::cpu::has_avx2()) {
        out.emplace_back("avx2", rabitqlib::simd::dot_l2_sqr_batch_avx2);
    }
    if (rabitqlib::cpu::has_avx512_core()) {
        out.emplace_back("avx512", rabitqlib::simd::dot_l2_sqr_batch_avx512);
    }
    return out;
}
}  // namespace

TEST(BatchDist, L2MatchesPerRowEuclidean) {
    for (size_t n : kRowCounts) {
        for (size_t dim : kDims) {
            auto mat = RandomMat(n, dim, 1);
            auto vec = RandomMat(1, dim, 2);
            std::vector<float> got(n);
            for (auto [name, fn] : L2Kernels()) {
                fn(mat.data(), vec.data(), n, dim, got.data());
                for (size_t i = 0; i < n; ++i) {
                    float want = euclidean_sqr(vec.data(), &mat[i * dim], dim);
                    EXPECT_NEAR(got[i], want, std::fabs(want) * 1e-5F + 1e-5F)
                        << name << " n=" << n << " dim=" << dim << " row=" << i;
                }
            }
        }
    }
}

TEST(BatchDist, DotL2MatchesPerRowPair) {
    for (size_t n : kRowCounts) {
        for (size_t dim : kDims) {
            auto mat = RandomMat(n, dim, 3);
            auto vec = RandomMat(1, dim, 4);
            std::vector<float> dots(n), dists(n);
            for (auto [name, fn] : DotL2Kernels()) {
                fn(mat.data(), vec.data(), n, dim, dots.data(), dists.data());
                for (size_t i = 0; i < n; ++i) {
                    float want_ip = dot_product(vec.data(), &mat[i * dim], dim);
                    float want_l2 = euclidean_sqr(vec.data(), &mat[i * dim], dim);
                    EXPECT_NEAR(dots[i], want_ip, std::fabs(want_ip) * 1e-5F + 1e-5F)
                        << name << " ip n=" << n << " dim=" << dim << " row=" << i;
                    EXPECT_NEAR(dists[i], want_l2, std::fabs(want_l2) * 1e-5F + 1e-5F)
                        << name << " l2 n=" << n << " dim=" << dim << " row=" << i;
                }
            }
        }
    }
}

// The reason for computing (a-b)^2 directly instead of expanding it into
// ||a||^2 - 2<a,b> + ||b||^2: on near-duplicate rows the expansion subtracts two
// large numbers to reach a small one and loses most of the significant digits,
// and can even land below zero. The direct form has no such failure mode, so
// this holds it to a tight bound with no clamping anywhere.
TEST(BatchDist, L2ExactOnNearDuplicateRows) {
    const size_t n = 23, dim = 960;
    auto vec = RandomMat(1, dim, 5);
    std::vector<float> mat(n * dim);
    std::mt19937 gen(6);
    std::normal_distribution<float> jitter(0.F, 1e-3F);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            mat[i * dim + d] = vec[d] + jitter(gen);
        }
    }
    std::vector<float> got(n);
    for (auto [name, fn] : L2Kernels()) {
        fn(mat.data(), vec.data(), n, dim, got.data());
        for (size_t i = 0; i < n; ++i) {
            float want = euclidean_sqr(vec.data(), &mat[i * dim], dim);
            EXPECT_GE(got[i], 0.0F) << name << " row " << i;
            EXPECT_NEAR(got[i], want, std::fabs(want) * 1e-5F) << name << " row " << i;
        }
    }
}
