#pragma once

namespace rabitqlib::cpu {

struct Features {
    bool avx2 = false;
    bool fma = false;
    bool avx512f = false;
    bool avx512bw = false;
    bool avx512dq = false;
    bool avx512vpopcntdq = false;
    bool avx512vnni = false;
};

const Features& features();
bool has_avx2();
bool has_avx512_core();
bool has_avx512_popcnt();
bool has_avx512_vnni();

}  // namespace rabitqlib::cpu
