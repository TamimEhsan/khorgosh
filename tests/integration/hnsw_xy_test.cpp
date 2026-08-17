#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/hnsw/hnsw.hpp"

using namespace rabitqlib;
using namespace rabitqlib::hnsw;

namespace {

constexpr size_t kDim = 128;
constexpr size_t kNumPoints = 1500;
constexpr size_t kNumQueries = 30;
// Small dataset for tests that only care about save/load plumbing, not
// recall/graph quality -- deliberately far from kNumPoints to reduce the
// odds of tripping a pre-existing, unrelated construction bug (see
// xy_quantization_plan.md's "known pre-existing HNSW issues" note).
constexpr size_t kSmallNumPoints = 200;

struct SyntheticData {
    std::vector<float> base;
    std::vector<float> queries;
    std::vector<float> centroid;
    std::vector<PID> cluster_ids;
};

SyntheticData MakeData(unsigned seed, size_t num_points = kNumPoints) {
    SyntheticData d;
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0F, 1.0F);

    d.base.resize(num_points * kDim);
    for (auto& x : d.base) {
        x = dist(gen);
    }
    d.queries.resize(kNumQueries * kDim);
    for (auto& x : d.queries) {
        x = dist(gen);
    }

    d.centroid.assign(kDim, 0.0F);
    for (size_t i = 0; i < num_points; ++i) {
        for (size_t j = 0; j < kDim; ++j) {
            d.centroid[j] += d.base[(i * kDim) + j];
        }
    }
    for (auto& c : d.centroid) {
        c /= static_cast<float>(num_points);
    }

    d.cluster_ids.assign(num_points, 0);
    return d;
}

std::vector<std::vector<PID>> BruteForceTopK(const SyntheticData& d, size_t k) {
    std::vector<std::vector<PID>> result(kNumQueries);
    for (size_t qi = 0; qi < kNumQueries; ++qi) {
        std::vector<std::pair<float, PID>> dists(kNumPoints);
        const float* q = &d.queries[qi * kDim];
        for (size_t i = 0; i < kNumPoints; ++i) {
            const float* p = &d.base[i * kDim];
            float s = 0;
            for (size_t j = 0; j < kDim; ++j) {
                float diff = q[j] - p[j];
                s += diff * diff;
            }
            dists[i] = {s, static_cast<PID>(i)};
        }
        std::partial_sort(dists.begin(), dists.begin() + static_cast<long>(k), dists.end());
        for (size_t i = 0; i < k; ++i) {
            result[qi].push_back(dists[i].second);
        }
    }
    return result;
}

float RecallAtK(
    const std::vector<std::vector<std::pair<float, PID>>>& results,
    const std::vector<std::vector<PID>>& ground_truth
) {
    size_t hits = 0;
    size_t total = 0;
    for (size_t qi = 0; qi < results.size(); ++qi) {
        std::vector<PID> got;
        got.reserve(results[qi].size());
        for (auto& p : results[qi]) {
            got.push_back(p.second);
        }
        for (PID gt : ground_truth[qi]) {
            if (std::find(got.begin(), got.end(), gt) != got.end()) {
                hits++;
            }
            total++;
        }
    }
    return static_cast<float>(hits) / static_cast<float>(total);
}

}  // namespace

// Backward-compat anchor for base_bits==1: the bit-for-bit equivalence of
// the new XyQuantBits{1, y} path vs. the old (total_bits) path is already
// verified exhaustively at the quantization level in
// tests/unit/rabitqlib/quantization/xy_quantization_test.cpp (same codes,
// same factors, same estimator formula). This test only exercises the new
// ctor's construct()+search() plumbing end to end; it deliberately does NOT
// also build via the *old* ctor+construct()+search() here, because that
// path hits a pre-existing, unrelated HNSW construction bug (misaligned
// uint64_t store in pack_binary(), see xy_quantization_plan.md) that's out
// of scope for this change and left alone per direction.
TEST(HnswXyIntegration, BaseBitsOneNewCtorConstructsAndSearches) {
    auto data = MakeData(1);

    HierarchicalNSW index(kNumPoints, kDim, XyQuantBits{1, 4}, 16, 100);
    EXPECT_EQ(index.base_bits(), 1U);
    EXPECT_EQ(index.nbits(), 5U);

    index.construct(
        1, data.centroid.data(), kNumPoints, data.base.data(), data.cluster_ids.data(), 1
    );
    auto results = index.search(data.queries.data(), kNumQueries, 10, 50, 1);

    auto gt = BruteForceTopK(data, 10);
    float recall = RecallAtK(results, gt);
    EXPECT_GT(recall, 0.5F);
}

TEST(HnswXyIntegration, BaseBitsGreaterThanOneGivesReasonableRecall) {
    auto data = MakeData(2);

    HierarchicalNSW index(kNumPoints, kDim, XyQuantBits{3, 4}, 16, 100);
    index.construct(
        1, data.centroid.data(), kNumPoints, data.base.data(), data.cluster_ids.data(), 1
    );
    auto results = index.search(data.queries.data(), kNumQueries, 10, 80, 1);

    auto gt = BruteForceTopK(data, 10);
    float recall = RecallAtK(results, gt);
    EXPECT_GT(recall, 0.5F);
}

TEST(HnswXyIntegration, SaveLoadRoundTripXyMode) {
    auto data = MakeData(3);
    const char* path = "hnsw_xy_test_index.bin";

    std::vector<std::vector<std::pair<float, PID>>> results_before;
    {
        HierarchicalNSW index(kNumPoints, kDim, XyQuantBits{3, 4}, 16, 100);
        index.construct(
            1, data.centroid.data(), kNumPoints, data.base.data(), data.cluster_ids.data(), 1
        );
        results_before = index.search(data.queries.data(), kNumQueries, 10, 80, 1);
        index.save(path);
    }

    HierarchicalNSW loaded;
    loaded.load(path);
    EXPECT_EQ(loaded.base_bits(), 3U);
    EXPECT_EQ(loaded.nbits(), 7U);
    auto results_after = loaded.search(data.queries.data(), kNumQueries, 10, 80, 1);

    ASSERT_EQ(results_before.size(), results_after.size());
    for (size_t qi = 0; qi < results_before.size(); ++qi) {
        ASSERT_EQ(results_before[qi].size(), results_after[qi].size());
        for (size_t i = 0; i < results_before[qi].size(); ++i) {
            EXPECT_EQ(results_before[qi][i].second, results_after[qi][i].second);
            EXPECT_NEAR(results_before[qi][i].first, results_after[qi][i].first, 1e-3F);
        }
    }

    std::remove(path);
}

TEST(HnswXyIntegration, LoadRejectsMismatchedFormatVersion) {
    auto data = MakeData(3, kSmallNumPoints);
    const char* path = "hnsw_xy_test_bad_version.bin";

    {
        HierarchicalNSW index(kSmallNumPoints, kDim, XyQuantBits{1, 2}, 16, 100);
        index.construct(
            1,
            data.centroid.data(),
            kSmallNumPoints,
            data.base.data(),
            data.cluster_ids.data(),
            1
        );
        index.save(path);
    }

    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(f.is_open());
        uint32_t bogus_version = 0xDEADBEEF;
        f.write(reinterpret_cast<const char*>(&bogus_version), sizeof(bogus_version));
    }

    HierarchicalNSW loaded;
    EXPECT_THROW(loaded.load(path), std::runtime_error);

    std::remove(path);
}
