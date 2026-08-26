#include "rabitqlib/utils/gemv.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "rabitqlib/utils/space.hpp"

using namespace rabitqlib;

namespace {

std::vector<float> RandomMat(size_t n, size_t dim, unsigned seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> nd(0.F, 1.F);
    std::vector<float> v(n * dim);
    for (auto& x : v) {
        x = nd(gen);
    }
    return v;
}

}  // namespace

TEST(Gemv, MatchesPerRowDotProduct) {
    for (size_t n : {1U, 7U, 64U, 512U}) {
        for (size_t dim : {64U, 128U, 960U}) {
            auto mat = RandomMat(n, dim, 10 + n + dim);
            auto vec = RandomMat(1, dim, 99);
            std::vector<float> got(n);
            gemv(mat.data(), vec.data(), n, dim, got.data());
            for (size_t i = 0; i < n; ++i) {
                float want = dot_product(&mat[i * dim], vec.data(), dim);
                ASSERT_NEAR(got[i], want, std::fabs(want) * 1e-4F + 1e-4F)
                    << "n=" << n << " dim=" << dim << " row " << i;
            }
        }
    }
}

TEST(Gemv, L2MatchesEuclideanSqr) {
    const size_t n = 64, dim = 960;
    auto mat = RandomMat(n, dim, 3);
    auto vec = RandomMat(1, dim, 4);
    std::vector<float> norms(n), dists(n), scratch(n);
    row_norms_sqr(mat.data(), n, dim, norms.data());
    l2_to_rows(mat.data(), norms.data(), vec.data(), n, dim, dists.data(), scratch.data());
    for (size_t i = 0; i < n; ++i) {
        float want = euclidean_sqr(&mat[i * dim], vec.data(), dim);
        ASSERT_NEAR(dists[i], want, std::fabs(want) * 1e-3F + 1e-3F) << "row " << i;
    }
}

// The expansion subtracts two large quantities to reach a small one. This is the
// case it is worst at: rows that sit almost on top of the query, so the true
// distance is a tiny fraction of the norms. The bound here is deliberately
// loose -- it documents how much precision the identity actually costs, rather
// than pretending it costs none.
TEST(Gemv, L2StaysNonNegativeUnderCancellation) {
    const size_t n = 32, dim = 960;
    auto vec = RandomMat(1, dim, 5);
    std::vector<float> mat(n * dim);
    std::mt19937 gen(6);
    std::normal_distribution<float> jitter(0.F, 1e-3F);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            mat[i * dim + d] = vec[d] + jitter(gen);
        }
    }
    std::vector<float> norms(n), dists(n), scratch(n);
    row_norms_sqr(mat.data(), n, dim, norms.data());
    l2_to_rows(mat.data(), norms.data(), vec.data(), n, dim, dists.data(), scratch.data());
    for (size_t i = 0; i < n; ++i) {
        float want = euclidean_sqr(&mat[i * dim], vec.data(), dim);
        EXPECT_GE(dists[i], 0.0F) << "row " << i;  // never negative, whatever else
        EXPECT_NEAR(dists[i], want, std::fabs(want) * 0.5F + 1e-3F)
            << "row " << i << " want " << want << " got " << dists[i];
    }
}

// l2_to_rows switches strategy on matrix size. Straddle kGemvL2ByteBudget so
// both paths run, and hold each to the same reference.
TEST(Gemv, L2AgreesAcrossTheSizeDispatch) {
    const size_t dim = 128;
    const size_t under = (kGemvL2ByteBudget / sizeof(float) / dim) / 2;
    const size_t over = (kGemvL2ByteBudget / sizeof(float) / dim) * 2;
    ASSERT_GT(under * dim * sizeof(float), 0UL);
    ASSERT_LE(under * dim * sizeof(float), kGemvL2ByteBudget);
    ASSERT_GT(over * dim * sizeof(float), kGemvL2ByteBudget);

    for (size_t n : {under, over}) {
        auto mat = RandomMat(n, dim, 7);
        auto vec = RandomMat(1, dim, 8);
        std::vector<float> norms(n), dists(n), scratch(n);
        row_norms_sqr(mat.data(), n, dim, norms.data());
        l2_to_rows(
            mat.data(), norms.data(), vec.data(), n, dim, dists.data(), scratch.data()
        );
        for (size_t i = 0; i < n; ++i) {
            float want = euclidean_sqr(&mat[i * dim], vec.data(), dim);
            EXPECT_NEAR(dists[i], want, std::fabs(want) * 1e-4F + 1e-4F)
                << "n=" << n << " row " << i;
        }
    }
}
