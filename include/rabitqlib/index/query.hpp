#pragma once

#include <cmath>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

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
        // For L2, dist is the compute by euclidean_sqr()
        // For IP, dist is computed by dot_product_dist() i.e., 1 - dot_product()
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

// Query-side prep for x+y quantization. No popcount/FastScan path here --
// both the base filter and the boosted refine step read their codes via
// generic SIMD dot kernels (see xy_base_estdist / xy_fulldist_boosting), so
// this just holds the rotated query and the two offset-binary corrections.
template <typename T>
class XYQuery {
   private:
    const T* rotated_query_;
    // The query scalar-quantized to kNumBits, one byte per dimension. Unlike
    // SplitSingleQuery this is NOT transposed into bit planes: the x+y kernels
    // consume it as bytes, against codes unpacked to bytes, via VNNI.
    std::vector<uint8_t> quant_query_;
    T delta_ = 0;
    T vl_ = 0;
    T G_add_ = 0;
    T G_error_ = 0;
    T G_kbxSumq_ = 0;
    T G_kbxySumq_ = 0;
    T G_kbxSumqHat_ = 0;
    T G_kbxySumqHat_ = 0;
    MetricType metric_type_ = METRIC_L2;

   public:
    // Carries both offset-binary corrections, since the two estimation
    // layers read codes of different widths: the base-only estimate needs
    // cb = -(2^base_bits - 1)/2, the combined estimate needs
    // cb = -(2^(base_bits+extra_bits) - 1)/2. One query object serves both.
    static constexpr size_t kNumBits = 4;

    explicit XYQuery(
        const T* rotated_query,
        size_t padded_dim,
        size_t base_bits,
        size_t extra_bits,
        MetricType metric_type = METRIC_L2,
        quant::RabitqConfig config = quant::RabitqConfig()
    )
        : rotated_query_(rotated_query), quant_query_(padded_dim, 0) {
        metric_type_ = (metric_type == METRIC_IP) ? METRIC_IP : METRIC_L2;

        float c_b_base = -static_cast<float>((1 << base_bits) - 1) / 2.F;
        float c_b = -static_cast<float>((1 << (base_bits + extra_bits)) - 1) / 2.F;
        T sumq =
            std::accumulate(rotated_query, rotated_query + padded_dim, static_cast<T>(0));

        G_kbxSumq_ = sumq * c_b_base;
        G_kbxySumq_ = sumq * c_b;

        // q_real ~= delta * q_int + vl, which is what lets a kernel recover the
        // real inner product from an integer one:
        //     sum(code * q_real) = delta * sum(code * q_int) + vl * sum(code)
        quant::quantize_scalar<float, uint8_t>(
            rotated_query, padded_dim, kNumBits, quant_query_.data(), delta_, vl_, config
        );

        // sum(q_hat) = delta * sum(q_int) + vl * dim, computed once per query so the
        // quantized estimators pay nothing for the correction at search time.
        T sum_qint = std::accumulate(
            quant_query_.begin(), quant_query_.end(), static_cast<T>(0)
        );
        T sumq_hat = (delta_ * sum_qint) + (vl_ * static_cast<T>(padded_dim));
        G_kbxSumqHat_ = sumq_hat * c_b_base;
        G_kbxySumqHat_ = sumq_hat * c_b;
    }

    [[nodiscard]] size_t num_bits() const { return kNumBits; }

    [[nodiscard]] const uint8_t* quant_query() const { return quant_query_.data(); }

    [[nodiscard]] T delta() const { return delta_; }

    [[nodiscard]] T vl() const { return vl_; }

    [[nodiscard]] const T* rotated_query() const { return rotated_query_; }

    [[nodiscard]] T kbxsumq() const { return G_kbxSumq_; }

    // Pair these with an inner product taken against quant_query(), never with one
    // taken against rotated_query().
    [[nodiscard]] T kbxsumq_hat() const { return G_kbxSumqHat_; }

    [[nodiscard]] T kbxysumq_hat() const { return G_kbxySumqHat_; }

    [[nodiscard]] T kbxysumq() const { return G_kbxySumq_; }

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
