// Side-by-side comparison of the two quantization schemes on ONE
// (query, centroid, vector) triple:
//
//   1. the exact L2 distance,
//   2. the legacy 1+y estimate  (1-bit filter -> ex-bit boosted refine),
//   3. the new x+y estimate     (x-bit filter -> y-bit boosted refine),
//
// so the same vector's error can be read off across bit budgets and, at
// equal total width, across split points.

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/space.hpp"

using namespace rabitqlib;

namespace {

// Shared inputs for every scheme below.
struct Problem {
    size_t dim;
    std::vector<float> vector;         // the database vector being quantized
    std::vector<float> centroid;       // its cluster centroid
    std::vector<float> rotated_query;  // the query, already rotated
    float actual_dist;                 // ||query - vector||^2
    float g_add;                       // ||query - centroid||^2
    float g_error;                     // ||query - centroid||
};

Problem MakeProblem(size_t dim, unsigned seed) {
    Problem p;
    p.dim = dim;
    p.vector.resize(dim);
    p.centroid.resize(dim);
    p.rotated_query.resize(dim);

    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.F, 1.F);
    for (size_t i = 0; i < dim; ++i) {
        p.centroid[i] = dist(gen);
    }
    for (size_t i = 0; i < dim; ++i) {
        p.rotated_query[i] = dist(gen);
    }
    for (size_t i = 0; i < dim; ++i) {
        p.vector[i] = dist(gen);
    }

    p.actual_dist = euclidean_sqr(p.rotated_query.data(), p.vector.data(), dim);
    float q_to_cent_sqr =
        euclidean_sqr(p.rotated_query.data(), p.centroid.data(), dim);
    p.g_add = q_to_cent_sqr;
    p.g_error = std::sqrt(q_to_cent_sqr);
    return p;
}

void PrintRow(
    const char* scheme, size_t total_bits, float est, float low, float actual
) {
    printf(
        "  %-12s %5zu  %14.3f  %14.3f  %+9.3f  %7.3f%%\n",
        scheme,
        total_bits,
        est,
        low,
        est - actual,
        100.F * std::fabs(est - actual) / actual
    );
}

// ---------------------------------------------------------------- 1 + y ---
// Legacy scheme: a 1-bit sign code with its own factors (BinDataMap) plus an
// ex_bits magnitude code (ExDataMap). The refine step boosts the 1-bit
// stage's inner product: (1<<ex_bits)*ip_x0_qr + ip(ex_code).
void RunLegacy(const Problem& p, size_t total_bits) {
    size_t ex_bits = total_bits - 1;

    std::vector<char> bin_data(BinDataMap<float>::data_bytes(p.dim));
    std::vector<char> ex_data(
        std::max<size_t>(ExDataMap<float>::data_bytes(p.dim, ex_bits), 1)
    );

    quant::quantize_split_single(
        p.vector.data(),
        p.centroid.data(),
        p.dim,
        ex_bits,
        bin_data.data(),
        ex_data.data(),
        METRIC_L2
    );

    quant::RabitqConfig config = quant::faster_config(p.dim, total_bits);
    SplitSingleQuery<float> query(
        p.rotated_query.data(), p.dim, ex_bits, config, METRIC_L2
    );

    // Filter stage: 1-bit code only.
    float ip_x0_qr = 0;
    float est = 0;
    float low = 0;
    split_single_estdist(
        bin_data.data(), query, p.dim, ip_x0_qr, est, low, p.g_add, p.g_error
    );
    if (ex_bits == 0) {
        // Nothing to refine, so the filter stage *is* the whole 1+0 scheme.
        PrintRow("legacy 1+0", 1, est, low, p.actual_dist);
        return;
    }
    PrintRow("legacy filt1", 1, est, low, p.actual_dist);

    // Refine stage: boosts the filter's inner product with the ex code.
    auto ip_func = select_excode_ipfunc(ex_bits);
    split_single_fulldist(
        bin_data.data(),
        ex_data.data(),
        ip_func,
        query,
        p.dim,
        ex_bits,
        est,
        low,
        ip_x0_qr,
        p.g_add,
        p.g_error
    );
    char label[32];
    snprintf(label, sizeof(label), "legacy 1+%zu", ex_bits);
    PrintRow(label, total_bits, est, low, p.actual_dist);
}

// ---------------------------------------------------------------- x + y ---
// New scheme: one (x+y)-bit code split into an x-bit base region (with its
// own factors) and a y-bit extra region. The refine step boosts the filter
// stage's base inner product: (1<<y)*ip_base + ip(extra_code).
void RunXy(const Problem& p, size_t base_bits, size_t extra_bits) {
    std::vector<char> base_data(BaseDataMap<float>::data_bytes(p.dim, base_bits));
    std::vector<char> extra_data(
        std::max<size_t>(ExDataMap<float>::data_bytes(p.dim, extra_bits), 1)
    );

    quant::quantize_xy_single(
        p.vector.data(),
        p.centroid.data(),
        p.dim,
        base_bits,
        extra_bits,
        base_data.data(),
        extra_data.data(),
        METRIC_L2
    );

    XYQuery<float> query(p.rotated_query.data(), p.dim, base_bits, extra_bits, METRIC_L2);

    // Filter stage: base code only.
    float ip_base = 0;
    float est = 0;
    float low = 0;
    xy_base_estdist(
        base_data.data(),
        select_excode_ipfunc(base_bits),
        query,
        p.dim,
        base_bits,
        ip_base,
        est,
        low,
        p.g_add,
        p.g_error
    );
    // Same filter stage, but with the query read as integers. Printed next to the
    // float version so the cost of quantizing the query is visible per width.
    float ip_base_q = 0;
    float est_q = 0;
    float low_q = 0;
    xy_base_estdist_quantized(
        base_data.data(),
        select_xy_base_ipfunc(),
        query,
        p.dim,
        base_bits,
        ip_base_q,
        est_q,
        low_q,
        p.g_add,
        p.g_error
    );

    char label[32];
    if (extra_bits == 0) {
        // No extra region, so the filter stage is the whole x+0 scheme.
        snprintf(label, sizeof(label), "xy %zu+0", base_bits);
        PrintRow(label, base_bits, est, low, p.actual_dist);
        snprintf(label, sizeof(label), "xyq %zu+0", base_bits);
        PrintRow(label, base_bits, est_q, low_q, p.actual_dist);
        return;
    }
    // Not the same code as a standalone x+0: this is the top x bits of an
    // (x+y)-bit code, so it is labelled as a stage, not as a scheme.
    snprintf(label, sizeof(label), "xy filt%zu", base_bits);
    PrintRow(label, base_bits, est, low, p.actual_dist);
    snprintf(label, sizeof(label), "xyq filt%zu", base_bits);
    PrintRow(label, base_bits, est_q, low_q, p.actual_dist);

    // Refine stage: reuses ip_base, reads only the extra region. Returns no
    // lower bound, exactly like the legacy split_distance_boosting.
    float boosted = xy_distance_boosting(
        extra_data.data(),
        select_excode_ipfunc(extra_bits),
        query,
        p.dim,
        extra_bits,
        ip_base,
        p.g_add
    );

    // One-shot path: recomputes ip_base itself and also yields a bound. Must
    // land on exactly the same estimate as the boosted path above.
    float full_est = 0;
    float full_low = 0;
    float full_ip_base = 0;
    xy_single_fulldist(
        base_data.data(),
        extra_data.data(),
        select_excode_ipfunc(base_bits),
        select_excode_ipfunc(extra_bits),
        query,
        p.dim,
        base_bits,
        extra_bits,
        full_est,
        full_low,
        full_ip_base,
        p.g_add,
        p.g_error
    );

    snprintf(label, sizeof(label), "xy %zu+%zu", base_bits, extra_bits);
    PrintRow(label, base_bits + extra_bits, full_est, full_low, p.actual_dist);

    if (full_est != boosted) {
        printf(
            "    !! boosted (%.6f) != one-shot (%.6f)\n",
            static_cast<double>(boosted),
            static_cast<double>(full_est)
        );
    }
}

}  // namespace

int main() {
    constexpr size_t kDim = 960;  // must be a multiple of 64
    Problem p = MakeProblem(kDim, 42);

    printf("dim = %zu\n", kDim);
    printf(
        "||query|| = %.4f  ||vector - centroid|| = %.4f  ||query - centroid|| = %.4f\n",
        static_cast<double>(std::sqrt(l2norm_sqr(p.rotated_query.data(), kDim))),
        static_cast<double>(
            std::sqrt(euclidean_sqr(p.vector.data(), p.centroid.data(), kDim))
        ),
        static_cast<double>(p.g_error)
    );
    printf("\nACTUAL DISTANCE  ||query - vector||^2 = %.4f\n", static_cast<double>(p.actual_dist));

    printf("\n  %-12s %5s  %14s  %14s  %9s  %8s\n",
           "scheme", "bits", "est_dist", "low_dist", "err", "rel_err");

    // Grouped by total width: the legacy 1+y scheme first, then every x+y split
    // of the same budget. The legacy row and the xy 1+y row directly below it
    // quantize the identical code, so they should agree -- that pairing is the
    // easiest thing to eyeball, which is why the splits run base-ascending.
    constexpr size_t kMaxTotalBits = 9;  // xy_bits::kMaxCombinedBits
    for (size_t total_bits = 1; total_bits <= kMaxTotalBits; ++total_bits) {
        printf(
            "\n-- %zu bit%s ------------------------------------------------------"
            "----\n",
            total_bits,
            total_bits == 1 ? "" : "s"
        );

        RunLegacy(p, total_bits);

        for (size_t base_bits = 1; base_bits <= total_bits; ++base_bits) {
            size_t extra_bits = total_bits - base_bits;
            if (base_bits > 8 || extra_bits > 8) {
                continue;  // each region is packed as at most 8 bits
            }
            RunXy(p, base_bits, extra_bits);
        }
    }

    return 0;
}
