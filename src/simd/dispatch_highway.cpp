// NOT an independent translation unit: #include-d by dispatch.cpp when
// RABITQLIB_TARGET_X86 is unset. Do not add this file to CMakeLists.txt's
// source list directly — doing so duplicate-defines every symbol below and
// fails the link.
//
// Phase 1/2 (see docs/portability/highway-plan.md): every public entry
// point dispatch_x86.cpp provides on x86 is mirrored here. Kernels that
// have a real `*_highway` implementation call it directly (currently: raw
// float space distances); kernels that only have a `*_generic` fallback so
// far call that; kernels with neither yet (rotation, sign flip, quantize,
// excode packing/transpose, FastScan accumulate, warmup, HNSW search) throw
// a descriptive error, the same way dispatch_x86.cpp does today on x86
// hardware without AVX2/AVX-512. Phase 2/3 upgrade these one at a time.
//
// Unlike dispatch_x86.cpp, there is no runtime tiering (no resolve_kernel)
// here: which implementation exists for a given function is a compile-time
// fact as this file is upgraded kernel by kernel, not something to probe at
// runtime. Once a kernel gets a `*_highway` implementation, Highway's own
// HWY_DYNAMIC_DISPATCH selects the best available target (SSE4/AVX3 on x86,
// NEON/SVE on ARM, WASM SIMD, RVV, or HWY_SCALAR) internally; this file
// never needs to choose between "highway" and "generic" itself.

#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/fastscan/fastscan.hpp"
#include "rabitqlib/fastscan/highacc_fastscan.hpp"
#include "rabitqlib/simd/dispatch.hpp"
#include "rabitqlib/simd/estimator_dispatch.hpp"
#include "rabitqlib/simd/fastscan_dispatch.hpp"
#include "rabitqlib/simd/hnsw_dispatch.hpp"
#include "rabitqlib/simd/matrix_dispatch.hpp"
#include "rabitqlib/simd/pack_excode_dispatch.hpp"
#include "rabitqlib/simd/quantization_dispatch.hpp"
#include "rabitqlib/simd/rotator_dispatch.hpp"
#include "rabitqlib/simd/space_dispatch.hpp"
#include "rabitqlib/simd/warmup_dispatch.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/warmup_space.hpp"

namespace rabitqlib::simd {

void matrix_product(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    matrix_product_generic(left, right, result, rows, inner, cols);
}

void matrix_product_transposed(
    const float* left,
    const float* right,
    float* result,
    size_t rows,
    size_t inner,
    size_t cols
) {
    matrix_product_transposed_generic(left, right, result, rows, inner, cols);
}

void row_norms(const float* data, float* result, size_t rows, size_t dim) {
    row_norms_generic(data, result, rows, dim);
}

void pairwise_distances_lower(
    const float* data,
    float* result,
    float* norms_data,
    size_t size,
    size_t dim,
    bool inner_product
) {
    pairwise_distances_lower_generic(data, result, norms_data, size, dim, inner_product);
}

void qg_batch_estdist(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance
) {
    qg_batch_estdist_generic(batch_data, q_obj, padded_dim, est_distance);
}

uint32_t qg_batch_estdist_mask(
    const char* batch_data,
    const BatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float threshold
) {
    return qg_batch_estdist_mask_generic(
        batch_data, q_obj, padded_dim, est_distance, threshold
    );
}

void split_batch_estdist(
    const char* batch_data,
    const SplitBatchQuery<float>& q_obj,
    size_t padded_dim,
    float* est_distance,
    float* low_distance,
    float* ip_x0_qr,
    bool use_hacc
) {
    split_batch_estdist_generic(
        batch_data, q_obj, padded_dim, est_distance, low_distance, ip_x0_qr, use_hacc
    );
}

float euclidean_sqr(const float* a, const float* b, size_t dim) {
    return euclidean_sqr_highway(a, b, dim);
}

float dot_product(const float* a, const float* b, size_t dim) {
    return dot_product_highway(a, b, dim);
}

float dot_product_dis(const float* a, const float* b, size_t dim) {
    return dot_product_dis_highway(a, b, dim);
}

float l2norm_sqr(const float* a, size_t dim) { return l2norm_sqr_highway(a, dim); }

[[noreturn]] static void missing_feature(const char* feature_name) {
    throw std::runtime_error(
        std::string(feature_name) +
        " is not yet implemented for the portable (non-x86) dispatch backend; "
        "see docs/portability/highway-plan.md"
    );
}

// With zero extra bits there is no extra code to contribute to the inner
// product, so the ex_bits == 0 slot must be a constant-zero stub rather than
// a duplicate of the 1-bit implementation. Matches dispatch_x86.cpp.
static float ip_fxu0(
    const float* /*query*/, const uint8_t* /*compact_code*/, size_t /*dim*/
) {
    return 0.0F;
}

double best_rescale_factor(
    const float* magnitudes, size_t dim, int max_code, double start, double end
) {
    return best_rescale_factor_generic(magnitudes, dim, max_code, start, end);
}

void fht_rotate(
    const float* data,
    float* rotated_vec,
    size_t dim,
    size_t padded_dim,
    size_t trunc_dim,
    float fac,
    const uint8_t* flip
) {
    fht_rotate_highway(data, rotated_vec, dim, padded_dim, trunc_dim, fac, flip);
}

static float missing_excode_ip(const float*, const uint8_t*, size_t) {
    missing_feature("excode ip functions");
}

ExcodeIpTable resolve_excode_ip_table() {
    return ExcodeIpTable{
        ip_fxu0,
        missing_excode_ip,
        missing_excode_ip,
        missing_excode_ip,
        missing_excode_ip,
        missing_excode_ip,
        missing_excode_ip,
        missing_excode_ip,
        missing_excode_ip,
    };
}

void flip_sign(const uint8_t* flip, float* data, size_t dim) {
    flip_sign_highway(flip, data, dim);
}

void kacs_walk(float* data, size_t len) { kacs_walk_highway(data, len); }

void scalar_quantize_uint8(
    uint8_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    scalar_quantize_uint8_highway(result, vec0, dim, lo, delta);
}

void scalar_quantize_uint16(
    uint16_t* result, const float* vec0, size_t dim, float lo, float delta
) {
    scalar_quantize_uint16_highway(result, vec0, dim, lo, delta);
}

static void missing_pack_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    (void)o_raw;
    (void)o_compact;
    (void)dim;
    missing_feature("excode packing");
}

void packing_2bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    missing_pack_excode(o_raw, o_compact, dim);
}

void packing_3bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    missing_pack_excode(o_raw, o_compact, dim);
}

void packing_4bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    missing_pack_excode(o_raw, o_compact, dim);
}

void packing_5bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    missing_pack_excode(o_raw, o_compact, dim);
}

void packing_6bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    missing_pack_excode(o_raw, o_compact, dim);
}

void packing_7bit_excode(const uint8_t* o_raw, uint8_t* o_compact, size_t dim) {
    missing_pack_excode(o_raw, o_compact, dim);
}

}  // namespace rabitqlib::simd

namespace rabitqlib {

const simd::ExcodeIpTable kExcodeIpTable = simd::resolve_excode_ip_table();

ex_ipfunc select_excode_ipfunc(size_t ex_bits) {
    if (ex_bits <= 8) {
        return kExcodeIpTable[ex_bits];
    }

    throw std::invalid_argument("Bad IP function for IVF");
}

float excode_ipimpl::ip16_fxu1_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kExcodeIpTable[1](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu2_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kExcodeIpTable[2](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu3_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kExcodeIpTable[3](query, compact_code, dim);
}

float excode_ipimpl::ip16_fxu4_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kExcodeIpTable[4](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu5_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kExcodeIpTable[5](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu6_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kExcodeIpTable[6](query, compact_code, dim);
}

float excode_ipimpl::ip64_fxu7_avx(
    const float* __restrict__ query, const uint8_t* __restrict__ compact_code, size_t dim
) {
    return kExcodeIpTable[7](query, compact_code, dim);
}

// Shared by this namespace and the rabitqlib::fastscan/rabitqlib::hnsw::detail
// blocks below, which reopen rabitqlib's nested namespaces later in this
// file and can therefore reach this via ordinary unqualified-name lookup;
// callers there qualify it explicitly (rabitqlib::missing_feature) anyway,
// to keep the dependency obvious regardless of declaration order.
[[noreturn]] static void missing_feature(const char* feature_name) {
    throw std::runtime_error(
        std::string(feature_name) +
        " is not yet implemented for the portable (non-x86) dispatch backend; "
        "see docs/portability/highway-plan.md"
    );
}

void new_transpose_bin(const uint16_t* q, uint64_t* tq, size_t padded_dim, size_t b_query) {
    (void)q;
    (void)tq;
    (void)padded_dim;
    (void)b_query;
    missing_feature("new transpose bin");
}

void new_transpose_bin_512(
    const uint8_t* q, uint64_t* tq, size_t padded_dim, size_t b_query
) {
    (void)q;
    (void)tq;
    (void)padded_dim;
    (void)b_query;
    missing_feature("new_transpose_bin_512");
}

float mask_ip_x0_q(const float* query, const uint8_t* data, size_t padded_dim) {
    (void)query;
    (void)data;
    (void)padded_dim;
    missing_feature("mask ip x0 q");
}

float mask_ip_x0_q(const float* query, const uint64_t* data, size_t padded_dim) {
    return mask_ip_x0_q(query, reinterpret_cast<const uint8_t*>(data), padded_dim);
}

}  // namespace rabitqlib

namespace rabitqlib::fastscan {

template <>
void pack_lut<float>(size_t dim, const float* __restrict__ query, float* __restrict__ lut) {
    simd::pack_lut_generic(dim, query, lut);
}

void accumulate(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ lp_table,
    int32_t* __restrict__ result,
    size_t dim
) {
    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument("FastScan dimension must be a positive multiple of 16");
    }
    (void)codes;
    (void)lp_table;
    (void)result;
    rabitqlib::missing_feature("fastscan accumulate");
}

void transfer_lut_hacc(const uint16_t* lut, size_t dim, uint8_t* hc_lut) {
    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument(
            "high-accuracy FastScan dimension must be a positive multiple of 16"
        );
    }
    (void)lut;
    (void)hc_lut;
    rabitqlib::missing_feature("fastscan high-accuracy LUT transfer");
}

void accumulate_hacc(
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ hc_lut,
    int32_t* accu_res,
    size_t dim
) {
    if (dim == 0 || dim % 16 != 0) {
        throw std::invalid_argument(
            "high-accuracy FastScan dimension must be a positive multiple of 16"
        );
    }
    (void)codes;
    (void)hc_lut;
    (void)accu_res;
    rabitqlib::missing_feature("fastscan high-accuracy accumulate");
}

}  // namespace rabitqlib::fastscan

namespace rabitqlib {

float warmup_ip_x0_q_512(
    const uint8_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    (void)data;
    (void)query;
    (void)delta;
    (void)vl;
    (void)padded_dim;
    (void)b_query;
    missing_feature("warmup_ip_x0_q_512");
}

float warmup_ip_x0_q_512(
    const uint64_t* data,
    const uint64_t* query,
    float delta,
    float vl,
    size_t padded_dim,
    size_t b_query
) {
    return warmup_ip_x0_q_512(
        reinterpret_cast<const uint8_t*>(data), query, delta, vl, padded_dim, b_query
    );
}

}  // namespace rabitqlib

namespace rabitqlib::hnsw::detail {

std::priority_queue<std::pair<float, PID>> search_knn(
    HierarchicalNSW& index, const float* query, size_t topk
) {
    (void)index;
    (void)query;
    (void)topk;
    rabitqlib::missing_feature("HNSW search");
}

}  // namespace rabitqlib::hnsw::detail
