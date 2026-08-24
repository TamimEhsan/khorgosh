#include "rabitqlib/simd/space_xy_dispatch.hpp"

#include <vector>

#include "rabitqlib/quantization/unpack_excode.hpp"

namespace rabitqlib::simd {

// Reference implementation, and the fallback where AVX512-VNNI is absent.
float xy_base_ip_u8_scalar(
    const uint8_t* quant_query,
    const uint8_t* code,
    size_t dim,
    size_t base_bits,
    float delta,
    float vl
) {
    std::vector<uint8_t> raw(dim);
    quant::rabitq_impl::ex_bits::unpacking_rabitqplus_code(code, raw.data(), dim, base_bits);

    uint64_t ip = 0;
    uint64_t sum = 0;
    for (size_t i = 0; i < dim; ++i) {
        ip += static_cast<uint64_t>(raw[i]) * quant_query[i];
        sum += raw[i];
    }
    return (delta * static_cast<float>(ip)) + (vl * static_cast<float>(sum));
}

}  // namespace rabitqlib::simd
