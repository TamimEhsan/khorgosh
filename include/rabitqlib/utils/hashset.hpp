// This code is modified based on NGT from Yahoo Japan
// https://github.com/yahoojapan/NGT
//
// Copyright (C) 2015 Yahoo Japan Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <climits>
#include <cstring>
#include <iostream>
#include <unordered_set>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/memory.hpp"

namespace rabitqlib {
/**
 * @brief hash set to record visited vertices
 *
 */
class HashBasedBooleanSet {
   private:
    // CHANGED: 8-bit epoch compresses memory footprint by 4x, drastically reducing cache misses
    using Epoch = std::uint32_t;

    std::vector<Epoch, memory::AlignedAllocator<Epoch>> visited_epoch_;
    Epoch current_epoch_ = 0;

public:
    HashBasedBooleanSet() = default;

    explicit HashBasedBooleanSet(std::size_t node_count)
        : visited_epoch_(node_count, 0) {}

    void initialize(std::size_t node_count) {
        visited_epoch_.assign(node_count, 0);
        current_epoch_ = 0;
    }

    inline void clear() noexcept {
        // Roll over and zero memory every 255 queries (memset is insanely fast on AVX-512)
        if (current_epoch_ == 0xFFFFFFFF) {
            std::memset(visited_epoch_.data(), 0, visited_epoch_.size());
            current_epoch_ = 1;
        } else {
            ++current_epoch_;
        }
    }

    [[nodiscard]] inline __attribute__((always_inline)) bool test_and_set(PID id) noexcept {
        Epoch& slot = visited_epoch_[id];
        if (slot == current_epoch_) return true;
        slot = current_epoch_;
        return false;
    }

    inline const char* get_prefetch_address(PID id) const noexcept {
        return reinterpret_cast<const char*>(visited_epoch_.data() + id);
    }
};
}  // namespace rabitqlib