#pragma once

#include <immintrin.h>
#include <omp.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/quantization/pack_excode.hpp"
#include "rabitqlib/utils/space.hpp"

namespace rabitqlib::quant::rabitq_impl {

constexpr float kConstEpsilon = 1.9;

namespace one_bit {

/**
 * @brief The one_bit_code function maps a data vector to a binary code by recording the
 * sign of every coordinate in an int array.
 *
 * @param data Input data vector to be quantized
 * @param centroid Input center vector used as reference for quantization
 * @param dim Dimensionality of the vectors
 * @param binary_code Output binary code where each element is 0 or 1 based on whether
 *                    the corresponding residual element (data - centroid) is positive
 *
 * @return RowMajorArray<T> The residual vector (data - centroid)
 *
 * This function computes a binary code by recording the sign of every coordinate
 * in the residual vector (data - centroid). It returns the residual vector for
 * further processing.
 */
template <typename T>
inline RowMajorArray<T> one_bit_code(
    const T* data, const T* centroid, size_t dim, int* binary_code
) {
    // map pointer to array
    ConstRowMajorArrayMap<T> data_arr(data, 1, dim);
    ConstRowMajorArrayMap<T> cent_arr(centroid, 1, dim);

    // residual vector
    RowMajorArray<T> residual_arr = data_arr - cent_arr;

    // unsigned representation
    RowMajorArrayMap<int> x_u(binary_code, 1, static_cast<long>(dim));
    x_u = (residual_arr > 0).template cast<int>();

    return residual_arr;
}

/**
 * @brief The one_bit_code_with_factor function maps a data vector to a binary code and
 * computes factors for distance estimation.
 *
 * @param data Input data vector to be quantized
 * @param centroid Input center vector used as reference for quantization
 * @param dim Dimensionality of the vectors
 * @param binary_code Output binary code where each element is 0 or 1 based on whether
 *                    the corresponding residual element (data - centroid) is positive
 * @param f_add Output factor used in distance estimation to add to the final result
 * @param f_rescale Output scaling factor used in distance estimation
 * @param f_error Output error bound for the distance estimation
 * @param metric_type Type of distance metric to use (L2 or inner product)
 *
 * This function computes a binary code by recording the sign of every coordinate
 * in the residual vector (data - centroid) and calculates the necessary factors
 * for accurate distance estimation. The factors are used in the estimator to
 * approximate distances between the original vectors.
 */
template <typename T>
inline void one_bit_code_with_factor(
    const T* data,
    const T* centroid,
    size_t dim,
    int* binary_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2
) {
    // quantize
    RowMajorArray<T> residual_arr = one_bit_code(data, centroid, dim, binary_code);

    // xu_cb = x_u + cb, xu_cb has same direction and different length with x_bar
    float cb = -((1 << 1) - 1) / 2.F;
    RowMajorArrayMap<int> x_u(binary_code, 1, static_cast<long>(dim));
    RowMajorArray<T> xu_cb = x_u.template cast<T>() + cb;

    // distance to centroid
    T l2_sqr = l2norm_sqr<T>(residual_arr.data(), dim);
    T l2_norm = std::sqrt(l2_sqr);

    // dot product between residual and xu_cb
    T ip_resi_xucb = dot_product<T>(residual_arr.data(), xu_cb.data(), dim);
    // dot product between centroid and xu_cb
    T ip_cent_xucb = dot_product<T>(centroid, xu_cb.data(), dim);

    // corner case
    if (ip_resi_xucb == 0) {
        ip_resi_xucb = std::numeric_limits<T>::infinity();
    }

    // We use unnormalized vector to get error factor. To be more specific,
    // sqrt((1 - <o, o_bar>^2) / <o, o_bar>^2) / sqrt(dim - 1) = 3rd item in following
    // expression
    T tmp_error =
        l2_norm * kConstEpsilon *
        std::sqrt(
            (((l2_sqr * l2norm_sqr<T>(xu_cb.data(), dim)) / (ip_resi_xucb * ip_resi_xucb)) -
             1) /
            (dim - 1)
        );

    // 3 factors used for distance estimation, please refer to document for more info.
    // For f_rescale and 2nd item of f_add, we use the dot product of raw residual (rather
    // than the normalized one) as the denominator, thus we need to multiply another l2norm.
    // For (ip_cent_xucb / ip_resi_xucb), the norm of xucb does not matter since it is also
    // in numerator.
    if (metric_type == METRIC_L2) {
        f_add = l2_sqr + (2 * l2_sqr * ip_cent_xucb / ip_resi_xucb);
        f_rescale = -2 * l2_sqr / ip_resi_xucb;
        f_error = 2 * tmp_error;
    } else if (metric_type == METRIC_IP) {
        f_add = 1 - dot_product<T>(residual_arr.data(), centroid, dim) +
                (l2_sqr * ip_cent_xucb / ip_resi_xucb);
        f_rescale = -l2_sqr / ip_resi_xucb;
        f_error = 1 * tmp_error;
    } else {
        std::cerr << "Unsupported metric type in quantization\n" << std::flush;
        exit(1);
    }
}

/**
 * @brief The one_bit_compact_code function maps a data vector to a compact binary code and
 * computes factors for distance estimation.
 *
 * @param data Input data vector to be quantized
 * @param centroid Input center vector used as reference for quantization
 * @param padded_dim Dimensionality of the vectors (padded to a multiple of the bit-packing
 * size)
 * @param compact_code Output compact binary code where the binary values are packed into
 * integers in big-endian order
 * @param f_add Output factor used in distance estimation to add to the final result
 * @param f_rescale Output scaling factor used in distance estimation
 * @param f_error Output error bound for the distance estimation
 * @param metric_type Type of distance metric to use (L2 or inner product)
 *
 * This function computes a compact binary code by recording the sign of every coordinate
 * in the residual vector (data - centroid) and packs the binary values into integers
 * for efficient storage and computation. It also calculates the necessary factors
 * for accurate distance estimation. The factors are used in the estimator to
 * approximate distances between the original vectors.
 */
template <typename T, typename TC>
inline void one_bit_compact_code(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    TC* compact_code,
    T& f_add,
    T& f_recale,
    T& f_error,
    MetricType metric_type = METRIC_L2
) {
    // binary code
    std::vector<int> binary_code(padded_dim);

    // get binary code
    one_bit_code_with_factor(
        data,
        centroid,
        padded_dim,
        binary_code.data(),
        f_add,
        f_recale,
        f_error,
        metric_type
    );

    pack_binary(binary_code.data(), compact_code, padded_dim);
}

// ! padded_dim % 64 == 0
template <typename T, typename TC, bool Parallel = false>
inline void one_bit_compact_codes(
    const T* data,
    const T* centroid,
    size_t num,
    size_t padded_dim,
    TC* compact_code,
    T* f_add,
    T* f_rescale,
    T* f_error,
    MetricType metric_type = METRIC_L2
) {
    constexpr size_t kTypeBits = sizeof(TC) * 8;

#pragma omp parallel for if (Parallel)
    for (size_t i = 0; i < num; ++i) {
        one_bit_compact_code(
            data + (padded_dim * i),
            centroid,
            padded_dim,
            compact_code + (padded_dim / kTypeBits * i),
            f_add[i],
            f_rescale[i],
            f_error[i],
            metric_type
        );
    }
}

// ! padded_dim % 64 == 0
template <typename T, bool Parallel = false>
inline void one_bit_batch_code(
    const T* data,
    const T* centroid,
    size_t num,
    size_t padded_dim,
    uint8_t* packed_code,
    T* f_add,
    T* f_recale,
    T* f_error,
    MetricType metric_type = METRIC_L2
) {
    std::vector<uint8_t> compact_codes(num * padded_dim / 8);

    one_bit_compact_codes<T, uint8_t, Parallel>(
        data,
        centroid,
        num,
        padded_dim,
        compact_codes.data(),
        f_add,
        f_recale,
        f_error,
        metric_type
    );
    fastscan::pack_codes(padded_dim, compact_codes.data(), num, packed_code);
}

}  // namespace one_bit

namespace ex_bits {
/**
 * @brief Get quantization code and factor (ipnorm_inv) for RaBitQ+
 * @note  Currently assume all number in o_prime are positive. For negative
 * values, the quantization code needs to be flipped.
 *
 * @param o_prime   Rotated & normalized data vector, length of dim
 * @param code      Quantization code of RaBitQ+, length of dim, stored as uint8
 * @param dim       Dimension of rotated vec
 * @param ex_bits   Number of bits used for code
 * @return ipnorm_inv  Returned factor
 */

constexpr std::array<float, 9> kTightStart = {
    0,
    0.15,
    0.20,
    0.52,
    0.59,
    0.71,
    0.75,
    0.77,
    0.81,
};

template <typename T>
inline double best_rescale_factor(const T* o_abs, size_t dim, size_t ex_bits) {
    constexpr double kEps = 1e-5;
    constexpr int kNEnum = 10;
    double max_o = *std::max_element(o_abs, o_abs + dim);

    double t_end = static_cast<double>(((1 << ex_bits) - 1) + kNEnum) / max_o;
    double t_start = t_end * kTightStart[ex_bits];

    std::vector<int> cur_o_bar(dim);
    double sqr_denominator = static_cast<double>(dim) * 0.25;
    double numerator = 0;

    for (size_t i = 0; i < dim; ++i) {
        int cur = static_cast<int>((t_start * o_abs[i]) + kEps);
        cur_o_bar[i] = cur;
        sqr_denominator += (cur * cur) + cur;
        numerator += (cur + 0.5) * o_abs[i];
    }

    std::priority_queue<
        std::pair<double, size_t>,
        std::vector<std::pair<double, size_t>>,
        std::greater<>>
        next_t;

    for (size_t i = 0; i < dim; ++i) {
        next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / o_abs[i], i);
    }

    double max_ip = 0;
    double t = 0;

    while (!next_t.empty()) {
        double cur_t = next_t.top().first;
        size_t update_id = next_t.top().second;
        next_t.pop();

        cur_o_bar[update_id]++;
        int update_o_bar = cur_o_bar[update_id];
        sqr_denominator += 2 * update_o_bar;
        numerator += o_abs[update_id];

        double cur_ip = numerator / std::sqrt(sqr_denominator);
        if (cur_ip > max_ip) {
            max_ip = cur_ip;
            t = cur_t;
        }

        if (update_o_bar < (1 << ex_bits) - 1) {
            double t_next = static_cast<double>(update_o_bar + 1) / o_abs[update_id];
            if (t_next < t_end) {
                next_t.emplace(t_next, update_id);
            }
        }
    }

    return t;
}

template <typename T, typename TP>
inline T quantize_ex(const T* o_abs, TP* code, size_t dim, size_t ex_bits) {
    constexpr double kEps = 1e-5;
    double t = best_rescale_factor<T>(o_abs, dim, ex_bits);
    double ipnorm = 0;

    std::vector<int> tmp_code(dim);
    for (size_t i = 0; i < dim; i++) {
        // compute and store code
        tmp_code[i] = static_cast<int>((t * o_abs[i]) + kEps);
        if (tmp_code[i] >= (1 << ex_bits)) {
            tmp_code[i] = (1 << ex_bits) - 1;
        }
        code[i] = static_cast<TP>(tmp_code[i]);

        // ip * norm = unnormalized ip
        ipnorm += (tmp_code[i] + 0.5) * o_abs[i];
    }

    T ipnorm_inv = static_cast<double>(1 / ipnorm);  // 1 / (ip*norm)
    if (!std::isnormal(ipnorm_inv)) {
        ipnorm_inv = 1.F;
    }

    return ipnorm_inv;
}

// For given dim and ex_bits, use random vectors to get the const rescale factor
inline double get_const_scaling_factors(size_t dim, size_t ex_bits) {
    constexpr long kConstNum = 100;

    RowMajorArray<double> rand = random_gaussian_matrix<double>(kConstNum, dim, 42);

    rand = rand.rowwise().normalized().abs();

    double sum = 0;
    for (long j = 0; j < kConstNum; ++j) {
        sum += best_rescale_factor(&rand(j, 0), dim, ex_bits);
    }

    double t_const = sum / kConstNum;

    return t_const;
}

template <typename T, typename TP>
T faster_quantize_ex(const T* o_abs, TP* code, size_t dim, size_t ex_bits, double t_const) {
    constexpr double kEps = 1e-5;
    double ipnorm = 0;

    std::vector<int> tmp_code(dim);
    for (size_t i = 0; i < dim; i++) {
        // compute and store code
        tmp_code[i] = static_cast<int>((t_const * o_abs[i]) + kEps);
        if (tmp_code[i] >= (1 << ex_bits)) {
            tmp_code[i] = (1 << ex_bits) - 1;
        }
        code[i] = static_cast<TP>(tmp_code[i]);

        // ip * norm = unnormalized ip
        ipnorm += (tmp_code[i] + 0.5) * o_abs[i];
    }

    T ipnorm_inv = static_cast<double>(1 / ipnorm);  // 1 / (ip*norm)
    if (!std::isnormal(ipnorm_inv)) {
        ipnorm_inv = 1.F;
    }

    return ipnorm_inv;
}

template <typename T, typename TP>
inline T ex_bits_code(
    const T* residual, size_t dim, size_t ex_bits, TP* ex_code, double t_const = -1
) {
    ConstRowMajorArrayMap<T> res_arr(residual, 1, dim);

    // get normalized abs residual for plus code
    RowMajorArray<T> abs_res = res_arr.rowwise().normalized().abs();

    // quantize data
    T ipnorm_inv = 1;
    if (t_const > 0) {
        ipnorm_inv = faster_quantize_ex(abs_res.data(), ex_code, dim, ex_bits, t_const);
    } else {
        ipnorm_inv = quantize_ex(abs_res.data(), ex_code, dim, ex_bits);
    }

    // revert codes for negative dims
    int32_t mask = (1 << ex_bits) - 1;
    for (size_t j = 0; j < dim; ++j) {
        if (res_arr.data()[j] < 0) {
            TP tmp = ex_code[j];
            ex_code[j] = (~tmp) & mask;
        }
    }

    return ipnorm_inv;
}

template <typename T, typename TP>
inline void ex_bits_code_with_factor(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t ex_bits,
    TP* ex_code,
    T& f_add_ex,
    T& f_rescale_ex,
    T& f_error_ex,
    MetricType metric_type = METRIC_L2,
    double t_const = -1
) {
    ConstRowMajorArrayMap<T> data_arr(data, 1, dim);
    ConstRowMajorArrayMap<T> cent_arr(centroid, 1, dim);

    // residual vector
    RowMajorArray<T> residual_arr = data_arr - cent_arr;

    T ipnorm_inv = ex_bits_code<T, TP>(residual_arr.data(), dim, ex_bits, ex_code, t_const);

    // get factors
    RowMajorArray<int> total_code =
        RowMajorArrayMap<TP>(ex_code, 1, dim).template cast<int>();
    for (size_t i = 0; i < dim; ++i) {
        total_code(0, static_cast<long>(i)) += static_cast<int>(residual_arr.data()[i] >= 0)
                                               << ex_bits;
    }

    // Factors are similar to those in one_bit_code_with_factor(), please refer to document
    // for detailed info.
    float cb = -(static_cast<float>(1 << ex_bits) - 0.5F);
    RowMajorArray<T> xu_cb = total_code.template cast<T>() + cb;

    T l2_sqr = l2norm_sqr(residual_arr.data(), dim);
    T l2_norm = std::sqrt(l2_sqr);

    T ip_resi_xucb = dot_product<T>(residual_arr.data(), xu_cb.data(), dim);
    T ip_cent_xucb = dot_product<T>(centroid, xu_cb.data(), dim);

    // corner case
    if (ip_resi_xucb == 0) {
        ip_resi_xucb = std::numeric_limits<T>::infinity();
    }

    T tmp_error =
        l2_norm * kConstEpsilon *
        std::sqrt(
            (((l2_sqr * l2norm_sqr<T>(xu_cb.data(), dim)) / (ip_resi_xucb * ip_resi_xucb)) -
             1) /
            (dim - 1)
        );

    if (metric_type == METRIC_L2) {
        f_add_ex = l2_sqr + (2 * l2_sqr * ip_cent_xucb / ip_resi_xucb);
        f_rescale_ex = ipnorm_inv * -2 * l2_norm;
        f_error_ex = 2 * tmp_error;
    } else if (metric_type == METRIC_IP) {
        f_add_ex = 1 - dot_product<T>(residual_arr.data(), centroid, dim) +
                   (l2_sqr * ip_cent_xucb / ip_resi_xucb);
        f_rescale_ex = ipnorm_inv * -l2_norm;
        f_error_ex = 1 * tmp_error;
    } else {
        std::cerr << "Unsupport metric for ex_bits_code()\n" << std::flush;
        exit(1);
    }
}

template <typename T, typename TP>
inline void ex_bits_compact_code(
    const T* data,
    const T* centroid,
    size_t padded_dim,
    size_t ex_bits,
    TP* compact_code,
    T& f_add_ex,
    T& f_rescale_ex,
    T& f_error_ex,
    MetricType metric_type = METRIC_L2,
    double t_const = -1
) {
    std::vector<uint8_t> ex_code(padded_dim);

    ex_bits_code_with_factor(
        data,
        centroid,
        padded_dim,
        ex_bits,
        ex_code.data(),
        f_add_ex,
        f_rescale_ex,
        f_error_ex,
        metric_type,
        t_const
    );

    packing_rabitqplus_code(ex_code.data(), compact_code, padded_dim, ex_bits);
}
}  // namespace ex_bits

namespace total_bits {
template <typename T, typename TP>
static inline void rabitq_scalar_impl(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& delta,
    T& vl,
    double t_const = -1,
    ScalarQuantizerType scalar_quantizer_type = ScalarQuantizerType::RECONSTRUCTION
) {
    std::vector<int> binary_code(dim);
    size_t ex_bits = total_bits - 1;

    RowMajorArray<T> residual_arr =
        rabitq_impl::one_bit::one_bit_code(data, centroid, dim, binary_code.data());

    if (ex_bits > 0) {
        ex_bits::ex_bits_code<T, TP>(
            residual_arr.data(), dim, ex_bits, total_code, t_const
        );
    }

    // merge 2 one_bit code and ex_bits code
    for (size_t i = 0; i < dim; ++i) {
        total_code[i] += static_cast<TP>(binary_code[i]) << ex_bits;
    }

    float cb = -(static_cast<float>(1 << ex_bits) - 0.5F);
    RowMajorArrayMap<TP> total_code_arr(total_code, 1, dim);
    RowMajorArray<T> u_cb = total_code_arr.template cast<T>() + cb;

    float norm_data = std::sqrt(l2norm_sqr(residual_arr.data(), dim));
    float norm_quan = std::sqrt(l2norm_sqr(u_cb.data(), dim));
    float cos_similarity =
        dot_product<T>(residual_arr.data(), u_cb.data(), dim) / (norm_data * norm_quan);

    if (scalar_quantizer_type == ScalarQuantizerType::RECONSTRUCTION) {
        delta = norm_data / norm_quan * cos_similarity;
    } else if (scalar_quantizer_type == ScalarQuantizerType::UNBIASED_ESTIMATION) {
        delta = norm_data / norm_quan / cos_similarity;
    } else if (scalar_quantizer_type == ScalarQuantizerType::PLAIN) {
        delta = norm_data / norm_quan;
    }

    vl = delta * cb;
}

static inline void rabitq_scalar_impl_avx512(
    const float* data,
    const float* centroid,
    size_t dim,
    size_t total_bits,
    uint8_t* total_code,
    float& delta,
    float& vl,
    double t_const = -1.0,
    ScalarQuantizerType scalar_quantizer_type = ScalarQuantizerType::RECONSTRUCTION
) {
    size_t ex_bits = total_bits - 1;
    float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);

    std::vector<float> residuals(dim);
    
    // ---------------------------------------------------------
    // PASS 1: Compute Residuals & Norms
    // ---------------------------------------------------------
    __m512 v_norm_data_sq = _mm512_setzero_ps();
    
    // Exact loop constraint - no scalar tail needed
    for (size_t i = 0; i < dim; i += 16) {
        // Note: If 'data' and 'centroid' are 64-byte aligned, 
        // change _mm512_loadu_ps to _mm512_load_ps
        __m512 v_d = _mm512_loadu_ps(&data[i]);
        __m512 v_c = _mm512_loadu_ps(&centroid[i]);
        
        __m512 v_res = _mm512_sub_ps(v_d, v_c);
        
        // _mm512_store_ps can be used here since std::vector<float> 
        // might not be strictly 64-byte aligned by default depending on the allocator, 
        // but loadu/storeu is perfectly safe and fast on modern architectures.
        _mm512_storeu_ps(&residuals[i], v_res);
        
        v_norm_data_sq = _mm512_fmadd_ps(v_res, v_res, v_norm_data_sq);
    }
    
    float norm_data_sq = _mm512_reduce_add_ps(v_norm_data_sq);
    float norm_data = std::sqrt(norm_data_sq);
    float inv_norm = (norm_data > 0.0f) ? (1.0f / norm_data) : 1.0f;

    // ---------------------------------------------------------
    // PASS 2: Fused Quantization, Coding, & Stats
    // ---------------------------------------------------------
    __m512 v_inv_norm = _mm512_set1_ps(inv_norm);
    __m512 v_t_const  = _mm512_set1_ps(static_cast<float>(t_const));
    __m512 v_kEps     = _mm512_set1_ps(1e-5f);
    __m512i v_mask_ex = _mm512_set1_epi32((1 << ex_bits) - 1);
    __m512 v_cb       = _mm512_set1_ps(cb);

    __m512 v_norm_quan_sq = _mm512_setzero_ps();
    __m512 v_dot_prod     = _mm512_setzero_ps();

    // Exact loop constraint - no scalar tail needed
    for (size_t i = 0; i < dim; i += 16) {
        __m512 v_res = _mm512_loadu_ps(&residuals[i]);

        __m512 v_abs_res    = _mm512_abs_ps(v_res);
        __m512 v_normed_abs = _mm512_mul_ps(v_abs_res, v_inv_norm);
        __m512 v_tmp_f      = _mm512_fmadd_ps(v_t_const, v_normed_abs, v_kEps);

        __m512i v_tmp_i = _mm512_cvttps_epi32(v_tmp_f); 
        v_tmp_i = _mm512_min_epi32(v_tmp_i, v_mask_ex);

        __mmask16 m_neg = _mm512_cmp_ps_mask(v_res, _mm512_setzero_ps(), _CMP_LT_OQ);
        __m512i v_not_tmp = _mm512_and_epi32(_mm512_xor_epi32(v_tmp_i, _mm512_set1_epi32(-1)), v_mask_ex);
        v_tmp_i = _mm512_mask_blend_epi32(m_neg, v_tmp_i, v_not_tmp);

        __mmask16 m_pos = _mm512_cmp_ps_mask(v_res, _mm512_setzero_ps(), _CMP_GT_OQ);
        __m512i v_bin_shifted = _mm512_maskz_set1_epi32(m_pos, 1 << ex_bits);

        __m512i v_tc_32 = _mm512_or_epi32(v_tmp_i, v_bin_shifted);

        // Pack down to uint8_t and store
        // Note: If total_code is 64-byte aligned, use _mm_store_si128
        __m128i v_tc_8 = _mm512_cvtepi32_epi8(v_tc_32);
        _mm_storeu_si128((__m128i*)&total_code[i], v_tc_8);

        __m512 v_tc_f = _mm512_cvtepi32_ps(v_tc_32);
        __m512 v_ucb  = _mm512_add_ps(v_tc_f, v_cb);

        v_norm_quan_sq = _mm512_fmadd_ps(v_ucb, v_ucb, v_norm_quan_sq);
        v_dot_prod     = _mm512_fmadd_ps(v_res, v_ucb, v_dot_prod);
    }

    float norm_quan_sq = _mm512_reduce_add_ps(v_norm_quan_sq);
    float dot_prod     = _mm512_reduce_add_ps(v_dot_prod);

    // ---------------------------------------------------------
    // Final Mathematics
    // ---------------------------------------------------------
    float norm_quan = std::sqrt(norm_quan_sq);
    float cos_similarity = dot_prod / (norm_data * norm_quan);

    if (scalar_quantizer_type == ScalarQuantizerType::RECONSTRUCTION) {
        delta = (norm_data / norm_quan) * cos_similarity;
    } else if (scalar_quantizer_type == ScalarQuantizerType::UNBIASED_ESTIMATION) {
        delta = (norm_data / norm_quan) / cos_similarity;
    } else if (scalar_quantizer_type == ScalarQuantizerType::PLAIN) {
        delta = norm_data / norm_quan;
    }

    vl = delta * cb;
}

template <typename T, typename TP>
static inline void rabitq_full_impl(
    const T* data,
    const T* centroid,
    size_t dim,
    size_t total_bits,
    TP* total_code,
    T& f_add,
    T& f_rescale,
    T& f_error,
    MetricType metric_type = METRIC_L2,
    double t_const = -1
) {
    std::vector<int> binary_code(dim);
    size_t ex_bits = total_bits - 1;

    rabitq_impl::one_bit::one_bit_code(data, centroid, dim, binary_code.data());

    if (ex_bits > 0) {
        ex_bits::ex_bits_code_with_factor<T, TP>(
            data,
            centroid,
            dim,
            ex_bits,
            total_code,
            f_add,
            f_rescale,
            f_error,
            metric_type,
            t_const
        );
    }

    for (size_t i = 0; i < dim; ++i) {
        total_code[i] += static_cast<TP>(binary_code[i]) << ex_bits;
    }
}
}  // namespace total_bits
}  // namespace rabitqlib::quant::rabitq_impl