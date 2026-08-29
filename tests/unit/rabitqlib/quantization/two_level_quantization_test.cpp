#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

namespace {

std::vector<float> RandomVec(size_t dim, std::mt19937& gen) {
    std::normal_distribution<float> dist(0.0F, 1.0F);
    std::vector<float> v(dim);
    for (auto& x : v) {
        x = dist(gen);
    }
    return v;
}

// pack_binary puts dimension j at bit (63 - j%64) of word j/64 -- the same
// convention new_transpose_bin_512 and mask_ip_x0_q use.
int PlaneBit(const uint64_t* plane, size_t j) {
    return static_cast<int>((plane[j / 64] >> (63 - (j % 64))) & 1ULL);
}

// Recover the raw per-dimension codes from a pack_excode block. A one-hot
// query makes the ex-code inner product return exactly code[j], so this reads
// the codes as the search path interprets them rather than reimplementing the
// bit layout. (Same trick as xy_quantization_test.cpp.)
std::vector<uint8_t> UnpackViaIpKernel(const uint8_t* packed, size_t dim, size_t bits) {
    std::vector<uint8_t> out(dim, 0);
    if (bits == 0) {
        return out;
    }
    auto ip = select_excode_ipfunc(bits);
    std::vector<float> probe(dim, 0.0F);
    for (size_t j = 0; j < dim; ++j) {
        probe[j] = 1.0F;
        out[j] = static_cast<uint8_t>(std::lround(ip(probe.data(), packed, dim)));
        probe[j] = 0.0F;
    }
    return out;
}

constexpr size_t kDim = 128;

// The estimator stages are templated on a Kernel so HNSW can bind its tuned
// per-ISA versions; here the runtime-dispatched ones are what we want.
struct DispatchKernel {
    static float warmup_ip_x0_q_512(
        const uint64_t* data,
        const uint64_t* query,
        float delta,
        float vl,
        size_t padded_dim,
        size_t b_query
    ) {
        return rabitqlib::warmup_ip_x0_q_512(data, query, delta, vl, padded_dim, b_query);
    }

    static float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
        return rabitqlib::mask_ip_x0_q(query, data, padded_dim);
    }
};

}  // namespace

// The claim the whole two-level design rests on: the high plane of the 2-bit
// base code is not an approximation of the 1-bit RaBitQ code, it *is* that
// code, byte for byte. If this ever stops holding, stage 1 of
// progressive_dist is estimating against something other than what it thinks.
TEST(TwoLevelQuantization, PlaneHiIsExactlyTheOneBitCode) {
    std::mt19937 gen(1234);

    for (size_t extra_bits = 0; extra_bits <= 8; ++extra_bits) {
        for (int trial = 0; trial < 8; ++trial) {
            std::vector<float> data = RandomVec(kDim, gen);
            std::vector<float> centroid = RandomVec(kDim, gen);

            std::vector<char> base_block(
                TwoBitBaseDataMap<float>::data_bytes(kDim), 0
            );
            std::vector<char> extra_block(
                extra_bits > 0 ? ExDataMap<float>::data_bytes(kDim, extra_bits) : 1, 0
            );
            quant::quantize_two_level_single(
                data.data(),
                centroid.data(),
                kDim,
                extra_bits,
                base_block.data(),
                extra_bits > 0 ? extra_block.data() : nullptr,
                METRIC_L2
            );

            std::vector<char> bin_block(BinDataMap<float>::data_bytes(kDim), 0);
            quant::quantize_compact_one_bit<float>(
                data.data(), centroid.data(), kDim, bin_block.data(), METRIC_L2
            );

            ConstTwoBitBaseDataMap<float> two_level(base_block.data(), kDim);
            ConstBinDataMap<float> one_bit(bin_block.data(), kDim);

            for (size_t w = 0; w < kDim / 64; ++w) {
                ASSERT_EQ(two_level.plane_hi()[w], one_bit.bin_code()[w])
                    << "extra_bits=" << extra_bits << " trial=" << trial
                    << " word=" << w;
            }
        }
    }
}

// The two planes must recombine to the same 2-bit base code the generic x+y
// path produces, and the factors must be identical -- same code, same
// derivation, only the storage differs.
TEST(TwoLevelQuantization, MatchesGenericXyPathAtBaseBitsTwo) {
    std::mt19937 gen(99);

    for (size_t extra_bits = 0; extra_bits <= 8; ++extra_bits) {
        std::vector<float> data = RandomVec(kDim, gen);
        std::vector<float> centroid = RandomVec(kDim, gen);

        std::vector<char> tl_base(TwoBitBaseDataMap<float>::data_bytes(kDim), 0);
        std::vector<char> tl_extra(
            extra_bits > 0 ? ExDataMap<float>::data_bytes(kDim, extra_bits) : 1, 0
        );
        quant::quantize_two_level_single(
            data.data(),
            centroid.data(),
            kDim,
            extra_bits,
            tl_base.data(),
            extra_bits > 0 ? tl_extra.data() : nullptr,
            METRIC_L2
        );

        std::vector<char> xy_base(BaseDataMap<float>::data_bytes(kDim, 2), 0);
        std::vector<char> xy_extra(
            extra_bits > 0 ? ExDataMap<float>::data_bytes(kDim, extra_bits) : 1, 0
        );
        quant::quantize_xy_single(
            data.data(),
            centroid.data(),
            kDim,
            2,
            extra_bits,
            xy_base.data(),
            extra_bits > 0 ? xy_extra.data() : nullptr,
            METRIC_L2
        );

        ConstTwoBitBaseDataMap<float> tl(tl_base.data(), kDim);
        ConstBaseDataMap<float> xy(xy_base.data(), kDim, 2);

        // Codes: 2*hi + lo == the packed 2-bit code.
        std::vector<uint8_t> xy_code = UnpackViaIpKernel(xy.base_code(), kDim, 2);
        for (size_t j = 0; j < kDim; ++j) {
            int recombined = (2 * PlaneBit(tl.plane_hi(), j)) + PlaneBit(tl.plane_lo(), j);
            ASSERT_EQ(recombined, static_cast<int>(xy_code[j]))
                << "extra_bits=" << extra_bits << " dim=" << j;
        }

        // Factors: bit-for-bit, not merely close -- same code_factors call.
        EXPECT_EQ(tl.f_add(), xy.f_add()) << "extra_bits=" << extra_bits;
        EXPECT_EQ(tl.f_rescale(), xy.f_rescale()) << "extra_bits=" << extra_bits;
        EXPECT_EQ(tl.f_error(), xy.f_error()) << "extra_bits=" << extra_bits;

        if (extra_bits > 0) {
            ConstExDataMap<float> tl_ex(tl_extra.data(), kDim, extra_bits);
            ConstExDataMap<float> xy_ex(xy_extra.data(), kDim, extra_bits);
            EXPECT_EQ(tl_ex.f_add_ex(), xy_ex.f_add_ex());
            EXPECT_EQ(tl_ex.f_rescale_ex(), xy_ex.f_rescale_ex());
            std::vector<uint8_t> a =
                UnpackViaIpKernel(tl_ex.ex_code(), kDim, extra_bits);
            std::vector<uint8_t> b =
                UnpackViaIpKernel(xy_ex.ex_code(), kDim, extra_bits);
            EXPECT_EQ(a, b) << "extra_bits=" << extra_bits;
        }
    }
}

// Layout contract: [plane_hi][3 factors][pad][plane_lo], both planes 8-byte
// aligned, stage 1's footprint the contiguous prefix.
TEST(TwoLevelQuantization, LayoutIsTwoPlanesAroundTheFactors) {
    for (size_t padded_dim : {64U, 128U, 960U, 1024U}) {
        EXPECT_EQ(
            TwoBitBaseDataMap<float>::data_bytes(padded_dim), (padded_dim / 4) + 24
        );

        std::vector<char> block(TwoBitBaseDataMap<float>::data_bytes(padded_dim), 0);
        TwoBitBaseDataMap<float> map(block.data(), padded_dim);

        auto base = reinterpret_cast<uintptr_t>(block.data());
        EXPECT_EQ(reinterpret_cast<uintptr_t>(map.plane_hi()) - base, 0U);
        EXPECT_EQ(
            reinterpret_cast<uintptr_t>(map.plane_lo()) - base, (padded_dim / 8) + 24
        );
        // Both planes must be 8-byte aligned relative to the block start,
        // since the kernels read them as uint64_t words.
        EXPECT_EQ(((padded_dim / 8) + 24) % 8, 0U);
        // plane_lo starts past the end of stage 1's footprint.
        EXPECT_GE(
            reinterpret_cast<uintptr_t>(map.plane_lo()) - base, (padded_dim / 8) + 12
        );
    }
}


namespace {

// Quantize `data` both ways and hand back the two blocks.
struct BothLayouts {
    std::vector<char> tl_base, tl_extra, xy_base, xy_extra;
};

BothLayouts QuantizeBothWays(
    const std::vector<float>& data, const std::vector<float>& centroid, size_t extra_bits
) {
    BothLayouts b;
    b.tl_base.assign(TwoBitBaseDataMap<float>::data_bytes(kDim), 0);
    b.tl_extra.assign(
        extra_bits > 0 ? ExDataMap<float>::data_bytes(kDim, extra_bits) : 1, 0
    );
    b.xy_base.assign(BaseDataMap<float>::data_bytes(kDim, 2), 0);
    b.xy_extra.assign(
        extra_bits > 0 ? ExDataMap<float>::data_bytes(kDim, extra_bits) : 1, 0
    );

    quant::quantize_two_level_single(
        data.data(),
        centroid.data(),
        kDim,
        extra_bits,
        b.tl_base.data(),
        extra_bits > 0 ? b.tl_extra.data() : nullptr,
        METRIC_L2
    );
    quant::quantize_xy_single(
        data.data(),
        centroid.data(),
        kDim,
        2,
        extra_bits,
        b.xy_base.data(),
        extra_bits > 0 ? b.xy_extra.data() : nullptr,
        METRIC_L2
    );
    return b;
}

}  // namespace

// Stage 2 reads the same code the generic x+y base estimator reads, through
// a different packing. The estimates must agree.
TEST(TwoLevelEstimator, Stage2MatchesGenericXyBaseEstimate) {
    constexpr size_t kExtraBits = 4;
    std::mt19937 gen(7);

    for (int trial = 0; trial < 32; ++trial) {
        std::vector<float> data = RandomVec(kDim, gen);
        std::vector<float> centroid = RandomVec(kDim, gen);
        std::vector<float> query = RandomVec(kDim, gen);
        auto b = QuantizeBothWays(data, centroid, kExtraBits);

        float norm = std::sqrt(euclidean_sqr(query.data(), centroid.data(), kDim));
        float g_add = norm * norm;
        float g_error = norm;

        XYQuery<float> xy_q(query.data(), kDim, 2, kExtraBits, METRIC_L2);
        TwoLevelQuery<float> tl_q(
            query.data(), kDim, kExtraBits, quant::RabitqConfig(), METRIC_L2
        );

        float ip_xy = 0;
        float est_xy = 0;
        float low_xy = 0;
        xy_base_estdist(
            b.xy_base.data(),
            select_excode_ipfunc(2),
            xy_q,
            kDim,
            2,
            ip_xy,
            est_xy,
            low_xy,
            g_add,
            g_error
        );

        float ip_tl = 0;
        float est_tl = 0;
        float low_tl = 0;
        two_level_estdist_2bit<DispatchKernel>(
            b.tl_base.data(), tl_q, kDim, ip_tl, est_tl, low_tl, g_add, g_error
        );

        ASSERT_NEAR(ip_tl, ip_xy, 1e-2F) << "trial=" << trial;
        ASSERT_NEAR(est_tl, est_xy, std::abs(est_xy) * 1e-4F + 1e-3F) << "trial=" << trial;
        ASSERT_NEAR(low_tl, low_xy, std::abs(low_xy) * 1e-4F + 1e-3F) << "trial=" << trial;
    }
}

// Stage 3 must reproduce the generic one-shot full x+y estimate.
TEST(TwoLevelEstimator, Stage3MatchesGenericXyFullEstimate) {
    std::mt19937 gen(11);

    for (size_t extra_bits : {1U, 2U, 4U, 7U, 8U}) {
        for (int trial = 0; trial < 16; ++trial) {
            std::vector<float> data = RandomVec(kDim, gen);
            std::vector<float> centroid = RandomVec(kDim, gen);
            std::vector<float> query = RandomVec(kDim, gen);
            auto b = QuantizeBothWays(data, centroid, extra_bits);

            float norm = std::sqrt(euclidean_sqr(query.data(), centroid.data(), kDim));
            float g_add = norm * norm;
            float g_error = norm;

            XYQuery<float> xy_q(query.data(), kDim, 2, extra_bits, METRIC_L2);
            TwoLevelQuery<float> tl_q(
                query.data(), kDim, extra_bits, quant::RabitqConfig(), METRIC_L2
            );

            float est_xy = 0;
            float low_xy = 0;
            float ip_xy = 0;
            xy_single_fulldist(
                b.xy_base.data(),
                b.xy_extra.data(),
                select_excode_ipfunc(2),
                select_excode_ipfunc(extra_bits),
                xy_q,
                kDim,
                2,
                extra_bits,
                est_xy,
                low_xy,
                ip_xy,
                g_add,
                g_error
            );

            float ip_tl = 0;
            float est2 = 0;
            float low2 = 0;
            two_level_estdist_2bit<DispatchKernel>(
                b.tl_base.data(), tl_q, kDim, ip_tl, est2, low2, g_add, g_error
            );
            ConstTwoBitBaseDataMap<float> tl_base(b.tl_base.data(), kDim);
            float est_tl = 0;
            float low_tl = 0;
            two_level_boosting(
                b.tl_extra.data(),
                select_excode_ipfunc(extra_bits),
                tl_q,
                kDim,
                extra_bits,
                ip_tl,
                tl_base.f_error(),
                est_tl,
                low_tl,
                g_add,
                g_error
            );

            ASSERT_NEAR(est_tl, est_xy, std::abs(est_xy) * 1e-4F + 1e-3F)
                << "extra_bits=" << extra_bits << " trial=" << trial;
            // Stage 3 now produces the refined bound too, and it is the same
            // one xy_single_fulldist derives: base f_error / 2^extra_bits.
            ASSERT_NEAR(low_tl, low_xy, std::abs(low_xy) * 1e-4F + 1e-3F)
                << "extra_bits=" << extra_bits << " trial=" << trial;
        }
    }
}

// The D1 risk, measured directly. Stage 1 reuses the 2-bit layer's f_error
// (widened by kTwoLevelStage1ErrorWidening) as a bound on an estimate that
// substituted the mean 1/2 for the unread low plane. That is a heuristic,
// not a derivation, so this pins how often it is wrong: every violation --
// low_dist above the true distance -- is a candidate the search would prune
// and should not have.
//
// Stage 2's rate is the control: the same f_error, used as designed.
TEST(TwoLevelEstimator, Stage1BoundViolationRateStaysNearStage2) {
    constexpr size_t kExtraBits = 4;
    constexpr int kTrials = 4000;
    std::mt19937 gen(2026);

    int violations_1 = 0;
    int violations_2 = 0;

    for (int trial = 0; trial < kTrials; ++trial) {
        std::vector<float> centroid = RandomVec(kDim, gen);
        // Data near the centroid, as a clustered index would have it.
        std::vector<float> data = RandomVec(kDim, gen);
        for (size_t j = 0; j < kDim; ++j) {
            data[j] = centroid[j] + (0.35F * data[j]);
        }
        std::vector<float> query = RandomVec(kDim, gen);

        auto b = QuantizeBothWays(data, centroid, kExtraBits);

        float norm = std::sqrt(euclidean_sqr(query.data(), centroid.data(), kDim));
        float g_add = norm * norm;
        float g_error = norm;
        float true_dist = euclidean_sqr(query.data(), data.data(), kDim);

        TwoLevelQuery<float> tl_q(
            query.data(), kDim, kExtraBits, quant::RabitqConfig(), METRIC_L2
        );

        float est1 = 0;
        float low1 = 0;
        two_level_estdist_1bit<DispatchKernel>(
            b.tl_base.data(), tl_q, kDim, est1, low1, g_add, g_error
        );
        float ip = 0;
        float est2 = 0;
        float low2 = 0;
        two_level_estdist_2bit<DispatchKernel>(
            b.tl_base.data(), tl_q, kDim, ip, est2, low2, g_add, g_error
        );

        violations_1 += static_cast<int>(low1 > true_dist);
        violations_2 += static_cast<int>(low2 > true_dist);
    }

    float rate_1 = static_cast<float>(violations_1) / kTrials;
    float rate_2 = static_cast<float>(violations_2) / kTrials;
    std::cerr << "[   INFO   ] stage-1 bound violation rate: " << rate_1
              << "  stage-2 (control): " << rate_2 << '\n';

    // Stage 1 is allowed to be looser than stage 2, but not wildly so --
    // this is the number that governs whether the early exit costs recall.
    EXPECT_LT(rate_1, 0.10F);
}

// The other half of the D1 question. A bound that is merely safe is worth
// nothing if it never fires -- the whole point of stage 1 is to answer
// "cannot enter the result set" without reading plane_lo or the extra
// region. This measures how often it actually does, against a threshold
// standing in for the search loop's distk (the k-th best distance seen so
// far), and reports the resulting stage histogram.
//
// Reported, not asserted beyond a floor: the real number depends on the
// dataset and on ef, and belongs in an end-to-end benchmark. This is here so
// a change that quietly makes stage 1 useless shows up in the test log.
TEST(TwoLevelEstimator, Stage1PrunesAMeaningfulShareOfCandidates) {
    constexpr size_t kExtraBits = 4;
    constexpr size_t kNumVectors = 2000;
    constexpr size_t kTopK = 10;
    constexpr int kQueries = 20;
    std::mt19937 gen(4242);

    std::vector<float> centroid = RandomVec(kDim, gen);
    std::vector<std::vector<char>> bases(kNumVectors);
    std::vector<std::vector<float>> vecs(kNumVectors);
    for (size_t i = 0; i < kNumVectors; ++i) {
        vecs[i] = RandomVec(kDim, gen);
        for (size_t j = 0; j < kDim; ++j) {
            vecs[i][j] = centroid[j] + (0.35F * vecs[i][j]);
        }
        std::vector<char> extra(ExDataMap<float>::data_bytes(kDim, kExtraBits), 0);
        bases[i].assign(TwoBitBaseDataMap<float>::data_bytes(kDim), 0);
        quant::quantize_two_level_single(
            vecs[i].data(),
            centroid.data(),
            kDim,
            kExtraBits,
            bases[i].data(),
            extra.data(),
            METRIC_L2
        );
    }

    size_t stop_at_1 = 0;
    size_t reached_2 = 0;

    for (int q = 0; q < kQueries; ++q) {
        std::vector<float> query = RandomVec(kDim, gen);
        float norm = std::sqrt(euclidean_sqr(query.data(), centroid.data(), kDim));
        float g_add = norm * norm;
        float g_error = norm;

        // Threshold: the true k-th nearest distance, which is what distk
        // converges to once the result set is full.
        std::vector<float> dists(kNumVectors);
        for (size_t i = 0; i < kNumVectors; ++i) {
            dists[i] = euclidean_sqr(query.data(), vecs[i].data(), kDim);
        }
        std::nth_element(dists.begin(), dists.begin() + kTopK, dists.end());
        float threshold = dists[kTopK];

        TwoLevelQuery<float> tl_q(
            query.data(), kDim, kExtraBits, quant::RabitqConfig(), METRIC_L2
        );

        for (size_t i = 0; i < kNumVectors; ++i) {
            float est1 = 0;
            float low1 = 0;
            two_level_estdist_1bit<DispatchKernel>(
                bases[i].data(), tl_q, kDim, est1, low1, g_add, g_error
            );
            if (low1 >= threshold) {
                ++stop_at_1;
            } else {
                ++reached_2;
            }
        }
    }

    float share = static_cast<float>(stop_at_1) /
                  static_cast<float>(stop_at_1 + reached_2);
    std::cerr << "[   INFO   ] stage-1 early exits: " << (100.0F * share)
              << "% of candidates (" << stop_at_1 << " of "
              << (stop_at_1 + reached_2) << ")\n";

    // Sanity floor only: if this ever drops to ~0, stage 1 is pure overhead
    // and the progressive structure is not earning its keep.
    EXPECT_GT(share, 0.10F);
}
