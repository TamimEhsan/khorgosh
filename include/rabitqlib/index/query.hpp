#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/lut.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib {
/**
 * @brief use an object to store data used for searching on symphonyqg for a given query
 */
template <typename T>
class BatchQuery {
   private:
    Lut<T> lookup_table_;
    T G_add_ = 0;
    T G_k1xSumq_ = 0;  // G_k1xSumq

   public:
    explicit BatchQuery(const T* rotated_query, size_t padded_dim) {
        lookup_table_ = std::move(Lut<T>(rotated_query, padded_dim));

        float c_1 = -((1 << 1) - 1) / 2.F;

        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
    }

    [[nodiscard]] T delta() const { return lookup_table_.delta(); }

    [[nodiscard]] T sum_vl_lut() const { return lookup_table_.sum_vl(); }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    void set_g_add(T dist) {
        // For L2, dist is computed by euclidean_sqr().
        // For IP, dist is computed by dot_product_dis(), i.e. 1 - dot_product().
        G_add_ = dist;
    }

    [[nodiscard]] const uint8_t* lut() const { return lookup_table_.lut(); }
};

template <typename T>
class SplitBatchQuery {
   private:
    const T* rotated_query_;
    Lut<T> lookup_table_;
    T G_add_ = 0;
    T G_error_ = 0;
    T G_k1xSumq_ = 0;
    T G_kbxSumq_ = 0;
    MetricType metric_type_ = METRIC_L2;

   public:
    explicit SplitBatchQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        MetricType metric_type = METRIC_L2,
        bool use_hacc = true
    )
        : rotated_query_(rotated_query) {
        lookup_table_ = std::move(Lut<T>(rotated_query, padded_dim, use_hacc));

        metric_type_ = (metric_type == METRIC_IP) ? METRIC_IP : METRIC_L2;

        float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
        float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
        G_kbxSumq_ = sumq * c_b;
    }
    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T delta() const { return lookup_table_.delta(); }

    [[nodiscard]] T sum_vl_lut() const { return lookup_table_.sum_vl(); }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    void set_g_add(T norm, T ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = norm * norm;
            G_error_ = norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -ip;
            G_error_ = norm;
        }
    }

    [[nodiscard]] const uint8_t* lut() const { return lookup_table_.lut(); }
};

template <typename T>
class SplitSingleQuery {
   private:
    const T* rotated_query_;
    std::vector<uint64_t> QueryBin_;
    T G_add_;
    T G_k1xSumq_;
    T G_kbxSumq_;
    T G_error_;
    T delta_;
    T vl_;
    MetricType metric_type_ = METRIC_L2;

   public:
    static constexpr size_t kNumBits = 4;
    explicit SplitSingleQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t ex_bits,
        quant::RabitqConfig config,
        size_t metric_type = METRIC_L2
    )
        : rotated_query_(rotated_query), QueryBin_(padded_dim * kNumBits / 64, 0) {
        float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
        float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_k1xSumq_ = sumq * c_1;
        G_kbxSumq_ = sumq * c_b;

        metric_type_ = (metric_type == METRIC_IP) ? METRIC_IP : METRIC_L2;

        std::vector<uint8_t> quant_query(padded_dim);

        // quantize query by rabitq
        quant::quantize_scalar<float, uint8_t>(
            rotated_query, padded_dim, kNumBits, quant_query.data(), delta_, vl_, config
        );

        // represent quantized query as u64
        rabitqlib::new_transpose_bin_512(
            quant_query.data(), QueryBin_.data(), padded_dim, kNumBits
        );

        // new_transpose_bin_512 already stores the query in the bit-plane/chunk
        // layout consumed by warmup_ip_x0_q_512.
    }

    [[nodiscard]] size_t num_bits() const { return kNumBits; }

    [[nodiscard]] const uint64_t* query_bin() const { return QueryBin_.data(); }

    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T delta() const { return delta_; }

    [[nodiscard]] T vl() const { return vl_; }

    [[nodiscard]] T k1xsumq() const { return G_k1xSumq_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    void set_g_add(T norm, T ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = norm * norm;
            G_error_ = norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -ip;
            G_error_ = norm;
        }
    }

    void set_g_error(T norm) { G_error_ = norm; }
};

// Query-side prep for the two-level (2+x) path. Deliberately a separate
// type from SplitSingleQuery rather than a generalization of it: the legacy
// 1+y path is the performance-critical one and must not shift, so it keeps
// its own class untouched. The machinery is the same -- the query is scalar
// quantized to kNumBits and transposed into bit planes, exactly so
// warmup_ip_x0_q_512 can consume it against the base code's high plane.
//
// What differs is the offset-binary corrections. Three code widths get
// estimated against over the course of one progressive_dist call, so three
// constants are carried:
//
//   k1xsumq()    stage 1, the 1-bit high plane        cb = -1/2
//   kbase_sumq() stage 2, the 2-bit base code         cb = -3/2
//   kbxsumq()    stage 3, the combined (2+x)-bit code cb = -(2^(2+x)-1)/2
//
// All three are just sumq scaled, so they cost one accumulate at query time.
template <typename T>
class TwoLevelQuery {
   private:
    const T* rotated_query_;
    std::vector<uint64_t> QueryBin_;
    T G_add_ = 0;
    T G_error_ = 0;
    T G_1xSumq_ = 0;
    T G_basexSumq_ = 0;
    T G_bxSumq_ = 0;
    T delta_ = 0;
    T vl_ = 0;
    MetricType metric_type_ = METRIC_L2;

   public:
    static constexpr size_t kNumBits = 4;
    static constexpr size_t kBaseBits = 2;

    explicit TwoLevelQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t extra_bits,
        quant::RabitqConfig config,
        size_t metric_type = METRIC_L2
    )
        : rotated_query_(rotated_query), QueryBin_(padded_dim * kNumBits / 64, 0) {
        const float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;
        const float c_base = -static_cast<float>((1 << kBaseBits) - 1) / 2.F;
        const float c_b = -static_cast<float>((1U << (kBaseBits + extra_bits)) - 1) / 2.F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_1xSumq_ = sumq * c_1;
        G_basexSumq_ = sumq * c_base;
        G_bxSumq_ = sumq * c_b;

        metric_type_ = (metric_type == METRIC_IP) ? METRIC_IP : METRIC_L2;

        std::vector<uint8_t> quant_query(padded_dim);

        quant::quantize_scalar<float, uint8_t>(
            rotated_query, padded_dim, kNumBits, quant_query.data(), delta_, vl_, config
        );

        rabitqlib::new_transpose_bin_512(
            quant_query.data(), QueryBin_.data(), padded_dim, kNumBits
        );
    }

    [[nodiscard]] size_t num_bits() const { return kNumBits; }

    [[nodiscard]] const uint64_t* query_bin() const { return QueryBin_.data(); }

    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T delta() const { return delta_; }

    [[nodiscard]] T vl() const { return vl_; }

    [[nodiscard]] T k1xsumq() const { return G_1xSumq_; }

    [[nodiscard]] T kbase_sumq() const { return G_basexSumq_; }

    [[nodiscard]] T kbxsumq() const { return G_bxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    void set_g_add(T norm, T ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = norm * norm;
            G_error_ = norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -ip;
            G_error_ = norm;
        }
    }
};

// Query-side prep for x+y quantization. No popcount/FastScan path here --
// both the base filter and the boosted refine step read their codes via
// generic SIMD dot kernels (see xy_base_estdist / xy_fulldist_boosting), so
// this just holds the rotated query and the two offset-binary corrections.
template <typename T>
class XYQuery {
   private:
    const T* rotated_query_;
    T G_add_ = 0;
    T G_error_ = 0;
    T G_kbxSumq_base_ = 0;
    T G_kbxSumq_ = 0;
    MetricType metric_type_ = METRIC_L2;

   public:
    // Carries both offset-binary corrections, since the two estimation
    // layers read codes of different widths: the base-only estimate needs
    // cb = -(2^base_bits - 1)/2, the combined estimate needs
    // cb = -(2^(base_bits+extra_bits) - 1)/2. One query object serves both.
    explicit XYQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t base_bits,
        size_t extra_bits,
        MetricType metric_type = METRIC_L2
    )
        : rotated_query_(rotated_query) {
        metric_type_ = (metric_type == METRIC_IP) ? METRIC_IP : METRIC_L2;

        float c_b_base = -static_cast<float>((1 << base_bits) - 1) / 2.F;
        float c_b = -static_cast<float>((1 << (base_bits + extra_bits)) - 1) / 2.F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_kbxSumq_base_ = sumq * c_b_base;
        G_kbxSumq_ = sumq * c_b;
    }

    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T kbxsumq_base() const { return G_kbxSumq_base_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    [[nodiscard]] T g_add() const { return G_add_; }

    [[nodiscard]] T g_error() const { return G_error_; }

    void set_g_add(T norm, T ip = 0) {
        if (metric_type_ == METRIC_L2) {
            G_add_ = norm * norm;
            G_error_ = norm;
        } else if (metric_type_ == METRIC_IP) {
            G_add_ = -ip;
            G_error_ = norm;
        }
    }
};

}  // namespace rabitqlib
