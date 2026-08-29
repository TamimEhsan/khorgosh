#pragma once

#include <cstdint>

#include "rabitqlib/fastscan/fastscan.hpp"

namespace rabitqlib {
template <typename T>
struct BatchDataMap {
   public:
    explicit BatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(reinterpret_cast<T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize)
        , f_error_(f_rescale_ + fastscan::kBatchSize) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] T* f_add() { return f_add_; }
    [[nodiscard]] T* f_rescale() { return f_rescale_; }
    [[nodiscard]] T* f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 3);
    }

   private:
    uint8_t* batch_bin_code_;
    T* f_add_;
    T* f_rescale_;
    T* f_error_;
};

template <typename T>
struct ConstBatchDataMap {
   public:
    explicit ConstBatchDataMap(const char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(reinterpret_cast<const T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize)
        , f_error_(f_rescale_ + fastscan::kBatchSize) {}

    [[nodiscard]] const uint8_t* bin_code() const { return batch_bin_code_; }
    [[nodiscard]] const T* f_add() const { return f_add_; }
    [[nodiscard]] const T* f_rescale() const { return f_rescale_; }
    [[nodiscard]] const T* f_error() const { return f_error_; }

   private:
    const uint8_t* batch_bin_code_;
    const T* f_add_;
    const T* f_rescale_;
    const T* f_error_;
};

template <typename T>
struct QGBatchDataMap {
   public:
    explicit QGBatchDataMap(char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(reinterpret_cast<T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize) {}

    [[nodiscard]] uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] T* f_add() { return f_add_; }
    [[nodiscard]] T* f_rescale() { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 2);
    }

   private:
    uint8_t* batch_bin_code_;
    T* f_add_;
    T* f_rescale_;
};

template <typename T>
struct ConstQGBatchDataMap {
   public:
    explicit ConstQGBatchDataMap(const char* data, size_t padded_dim)
        : batch_bin_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(reinterpret_cast<const T*>(data + (padded_dim * fastscan::kBatchSize / 8))
          )  // 1 bit code
        , f_rescale_(f_add_ + fastscan::kBatchSize) {}

    [[nodiscard]] const uint8_t* bin_code() { return batch_bin_code_; }
    [[nodiscard]] const T* f_add() { return f_add_; }
    [[nodiscard]] const T* f_rescale() { return f_rescale_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim * fastscan::kBatchSize / 8) +
               (sizeof(T) * fastscan::kBatchSize * 2);
    }

   private:
    const uint8_t* batch_bin_code_;
    const T* f_add_;
    const T* f_rescale_;
};

template <typename T>
struct ExDataMap {
   public:
    explicit ExDataMap(char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_ex_(*reinterpret_cast<T*>(data + (padded_dim * ex_bits / 8)))
        , f_recale_ex_(*(reinterpret_cast<T*>(data + (padded_dim * ex_bits / 8)) + 1)) {}

    static size_t data_bytes(size_t padded_dim, size_t ex_bits) {
        return ex_bits > 0 ? (padded_dim * ex_bits / 8) + (sizeof(T) * 2) : 0;
    }

    [[nodiscard]] uint8_t* ex_code() { return ex_code_; }
    [[nodiscard]] T& f_add_ex() { return f_add_ex_; }
    [[nodiscard]] T& f_rescale_ex() { return f_recale_ex_; }

   private:
    uint8_t* ex_code_;
    T& f_add_ex_;
    T& f_recale_ex_;
};

template <typename T>
struct ConstExDataMap {
   public:
    explicit ConstExDataMap(const char* data, size_t padded_dim, size_t ex_bits)
        : ex_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_ex_(*reinterpret_cast<const T*>(data + (padded_dim * ex_bits / 8)))
        , f_recale_ex_(*(reinterpret_cast<const T*>(data + (padded_dim * ex_bits / 8)) + 1)
          ) {}

    [[nodiscard]] const uint8_t* ex_code() const { return ex_code_; }
    [[nodiscard]] const T& f_add_ex() const { return f_add_ex_; }
    [[nodiscard]] const T& f_rescale_ex() const { return f_recale_ex_; }

   private:
    const uint8_t* ex_code_;
    const T& f_add_ex_;
    const T& f_recale_ex_;
};

// Base layer of a split-code vector: the top base_bits of the combined code,
// packed via pack_excode, plus its own three factors. This is the cheap filter
// layer, and it plays exactly the role BinDataMap plays in the 1+y layout --
// which is what it collapses to at base_bits == 1. The refine layer that pairs
// with it is ExDataMap: the bottom extra_bits of the *same* combined code
// (never a second copy of the base bits), plus that code's f_add/f_rescale.
template <typename T>
struct BaseDataMap {
   public:
    explicit BaseDataMap(char* data, size_t padded_dim, size_t base_bits)
        : base_code_(reinterpret_cast<uint8_t*>(data))
        , f_add_(*reinterpret_cast<T*>(data + (padded_dim * base_bits / 8)))
        , f_rescale_(*(reinterpret_cast<T*>(data + (padded_dim * base_bits / 8)) + 1))
        , f_error_(*(reinterpret_cast<T*>(data + (padded_dim * base_bits / 8)) + 2)) {}

    static size_t data_bytes(size_t padded_dim, size_t base_bits) {
        return (padded_dim * base_bits / 8) + (sizeof(T) * 3);
    }

    [[nodiscard]] uint8_t* base_code() { return base_code_; }
    [[nodiscard]] T& f_add() { return f_add_; }
    [[nodiscard]] T& f_rescale() { return f_rescale_; }
    [[nodiscard]] T& f_error() { return f_error_; }

   private:
    uint8_t* base_code_;
    T& f_add_;
    T& f_rescale_;
    T& f_error_;
};

template <typename T>
struct ConstBaseDataMap {
   public:
    explicit ConstBaseDataMap(const char* data, size_t padded_dim, size_t base_bits)
        : base_code_(reinterpret_cast<const uint8_t*>(data))
        , f_add_(*reinterpret_cast<const T*>(data + (padded_dim * base_bits / 8)))
        , f_rescale_(*(reinterpret_cast<const T*>(data + (padded_dim * base_bits / 8)) + 1))
        , f_error_(*(reinterpret_cast<const T*>(data + (padded_dim * base_bits / 8)) + 2)) {
    }

    [[nodiscard]] const uint8_t* base_code() const { return base_code_; }
    [[nodiscard]] const T& f_add() const { return f_add_; }
    [[nodiscard]] const T& f_rescale() const { return f_rescale_; }
    [[nodiscard]] const T& f_error() const { return f_error_; }

   private:
    const uint8_t* base_code_;
    const T& f_add_;
    const T& f_rescale_;
    const T& f_error_;
};

// Two-level (2+x) base region. Holds the 2-bit base code as two separate
// 1-bit planes rather than as a packed 2-bit excode, so each plane is
// directly consumable by the existing binary kernels: plane_hi is a valid
// 1-bit RaBitQ code (it is the sign bit of the combined code -- see
// xy_bits::two_level_split_code_with_factor) readable by
// warmup_ip_x0_q_512, and both planes are readable by mask_ip_x0_q. No new
// SIMD kernel is needed for either.
//
// Two factor triples, not one. plane_hi is a 1-bit code in its own right and
// gets the factors derived for a 1-bit code (cb = -1/2); the 2-bit code made
// of both planes gets its own (cb = -3/2). Sharing one triple across both
// widths does not work: f_rescale divides by <residual, code + cb>, which
// differs per width, and a stage-1 bound improvised from the 2-bit f_error
// over-prunes badly as dimension grows (measured: recall@10 0.73 vs 0.98 on
// gist at dim 960). The 12 extra bytes buy the same calibrated bound the
// 1+y path has always used.
//
// The factors sit *between* the planes on purpose: the cheap first stage
// needs plane_hi and the 1-bit triple and nothing else, so its footprint is
// the contiguous prefix [0, padded_dim/8 + 12). plane_lo is only pulled into
// cache if that stage declines to prune. Both planes stay 8-byte aligned
// (padded_dim/8 is a multiple of 8 because padded_dim % 64 == 0, and the two
// triples together are 24 bytes).
//
// Layout: [plane_hi][f_add_1 f_rescale_1 f_error_1][f_add f_rescale f_error][plane_lo]
template <typename T>
struct TwoBitBaseDataMap {
   public:
    explicit TwoBitBaseDataMap(char* data, size_t padded_dim)
        : plane_hi_(reinterpret_cast<uint64_t*>(data))
        , factors_(reinterpret_cast<T*>(data + (padded_dim / 8)))
        , plane_lo_(reinterpret_cast<uint64_t*>(data + (padded_dim / 8) + kFactorBytes)) {}

    static constexpr size_t kFactorBytes = 24;  // two triples of 3 floats

    static size_t data_bytes(size_t padded_dim) { return (padded_dim / 4) + kFactorBytes; }

    [[nodiscard]] uint64_t* plane_hi() { return plane_hi_; }
    [[nodiscard]] uint64_t* plane_lo() { return plane_lo_; }

    // 1-bit layer (plane_hi alone, cb = -1/2)
    [[nodiscard]] T& f_add_1() { return factors_[0]; }
    [[nodiscard]] T& f_rescale_1() { return factors_[1]; }
    [[nodiscard]] T& f_error_1() { return factors_[2]; }

    // 2-bit layer (both planes, cb = -3/2)
    [[nodiscard]] T& f_add() { return factors_[3]; }
    [[nodiscard]] T& f_rescale() { return factors_[4]; }
    [[nodiscard]] T& f_error() { return factors_[5]; }

   private:
    uint64_t* plane_hi_;
    T* factors_;
    uint64_t* plane_lo_;
};

template <typename T>
struct ConstTwoBitBaseDataMap {
   public:
    explicit ConstTwoBitBaseDataMap(const char* data, size_t padded_dim)
        : plane_hi_(reinterpret_cast<const uint64_t*>(data))
        , factors_(reinterpret_cast<const T*>(data + (padded_dim / 8)))
        , plane_lo_(reinterpret_cast<const uint64_t*>(
              data + (padded_dim / 8) + TwoBitBaseDataMap<T>::kFactorBytes
          )) {}

    static size_t data_bytes(size_t padded_dim) {
        return TwoBitBaseDataMap<T>::data_bytes(padded_dim);
    }

    [[nodiscard]] const uint64_t* plane_hi() const { return plane_hi_; }
    [[nodiscard]] const uint64_t* plane_lo() const { return plane_lo_; }

    [[nodiscard]] const T& f_add_1() const { return factors_[0]; }
    [[nodiscard]] const T& f_rescale_1() const { return factors_[1]; }
    [[nodiscard]] const T& f_error_1() const { return factors_[2]; }

    [[nodiscard]] const T& f_add() const { return factors_[3]; }
    [[nodiscard]] const T& f_rescale() const { return factors_[4]; }
    [[nodiscard]] const T& f_error() const { return factors_[5]; }

   private:
    const uint64_t* plane_hi_;
    const T* factors_;
    const uint64_t* plane_lo_;
};

template <typename T>
struct BinDataMap {
   public:
    explicit BinDataMap(char* data, size_t padded_dim)
        : bin_code_(reinterpret_cast<uint64_t*>(data))
        , f_add_(*reinterpret_cast<T*>(data + (padded_dim / 8)))
        , f_rescale_(*(reinterpret_cast<T*>(data + (padded_dim / 8)) + 1))
        , f_error_(*(reinterpret_cast<T*>(data + (padded_dim / 8)) + 2)) {}

    [[nodiscard]] uint64_t* bin_code() { return bin_code_; }
    [[nodiscard]] T& f_add() { return f_add_; }
    [[nodiscard]] T& f_rescale() { return f_rescale_; }
    [[nodiscard]] T& f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim / 8) + (sizeof(T) * 3);
    }

   private:
    uint64_t* bin_code_;
    T& f_add_;
    T& f_rescale_;
    T& f_error_;
};

template <typename T>
struct ConstBinDataMap {
   public:
    explicit ConstBinDataMap(const char* data, size_t padded_dim)
        : bin_code_(reinterpret_cast<const uint64_t*>(data))
        , f_add_(*reinterpret_cast<const T*>(data + (padded_dim / 8)))
        , f_rescale_(*(reinterpret_cast<const T*>(data + (padded_dim / 8)) + 1))
        , f_error_(*(reinterpret_cast<const T*>(data + (padded_dim / 8)) + 2)) {}

    [[nodiscard]] const uint64_t* bin_code() { return bin_code_; }
    [[nodiscard]] const T& f_add() { return f_add_; }
    [[nodiscard]] const T& f_rescale() { return f_rescale_; }
    [[nodiscard]] const T& f_error() { return f_error_; }

    static size_t data_bytes(size_t padded_dim) {
        return (padded_dim / 8) + (sizeof(T) * 3);
    }

   private:
    const uint64_t* bin_code_;
    const T& f_add_;
    const T& f_rescale_;
    const T& f_error_;
};
}  // namespace rabitqlib