// HNSW is developed from the [HNSW library](https://github.com/nmslib/hnswlib)
#pragma once

#include <omp.h>

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/estimator.hpp"
#include "rabitqlib/index/ivf/initializer.hpp"
#include "rabitqlib/index/query.hpp"
#include "rabitqlib/quantization/data_layout.hpp"
#include "rabitqlib/quantization/rabitq.hpp"
#include "rabitqlib/utils/buffer.hpp"
#include "rabitqlib/utils/cpu_features.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "rabitqlib/utils/visited_pool.hpp"

namespace rabitqlib::hnsw {

template <typename T>
using maxheap = std::priority_queue<T>;

template <typename T>
using minheap = std::priority_queue<T, std::vector<T>, std::greater<T>>;

class HierarchicalNSW;

// Tag type disambiguating the x+y constructor overload from the existing
// (max_elements, dim, total_bits, M, ef_construction, ...) constructor --
// a same-signature overload taking base_bits/extra_bits as plain size_t
// would collide with it under overload resolution.
struct XyQuantBits {
    size_t base_bits;
    size_t extra_bits;
};

namespace detail {

maxheap<std::pair<float, PID>> search_knn_avx2(HierarchicalNSW&, const float*, size_t);

maxheap<std::pair<float, PID>> search_knn_avx512_core(
    HierarchicalNSW&, const float*, size_t
);

maxheap<std::pair<float, PID>> search_knn_avx512_popcnt(
    HierarchicalNSW&, const float*, size_t
);

}  // namespace detail

class HierarchicalNSW {
   public:
    explicit HierarchicalNSW(){};
    explicit HierarchicalNSW(
        size_t, size_t, size_t, size_t, size_t, size_t = 100, MetricType = METRIC_L2
    );
    explicit HierarchicalNSW(
        size_t, size_t, XyQuantBits, size_t, size_t, size_t = 100, MetricType = METRIC_L2
    );
    ~HierarchicalNSW();

    [[nodiscard]] size_t dimension() const { return dim_; }
    [[nodiscard]] size_t num_clusters() const { return num_cluster_; }
    [[nodiscard]] size_t base_bits() const { return base_bits_; }
    [[nodiscard]] size_t nbits() const { return base_bits_ + ex_bits_; }
    [[nodiscard]] size_t M() const { return M_; }
    [[nodiscard]] size_t ef_construction() const { return ef_construction_; }
    [[nodiscard]] MetricType metric_type() const { return metric_type_; }
    [[nodiscard]] size_t max_elements() const { return max_elements_; }

    void save(const char*) const;
    void load(const char*);

    void construct(size_t, const float*, size_t, const float*, PID*, size_t, bool);
    std::vector<std::vector<std::pair<float, PID>>> search(
        const float*, size_t, size_t, size_t, size_t
    );

    const float* rawDataPtr_{nullptr};

    struct ResultRecord {
        float est_dist;
        float low_dist;
        ResultRecord(float est_dist, float low_dist)
            : est_dist(est_dist), low_dist(low_dist) {}
        bool operator<(const ResultRecord& other) const {
            return this->est_dist < other.est_dist;
        }
    };

    // Bounded priority queue implemented as a sorted vector.
    struct Candidate {
        HierarchicalNSW::ResultRecord record;
        PID id;
    };

    class BoundedKNN {
       public:
        explicit BoundedKNN(size_t capacity) : capacity_(capacity) {}

        // Insert a candidate in sorted order (ascending by est_dist).
        void insert(const Candidate& cand) {
            // Find insertion position using binary search.
            auto it = std::upper_bound(
                queue_.begin(),
                queue_.end(),
                cand,
                [](const Candidate& a, const Candidate& b) {
                    return a.record.est_dist < b.record.est_dist;
                }
            );
            queue_.insert(it, cand);
            // If we exceed capacity, drop the worst candidate (largest est_dist).
            if (queue_.size() > capacity_) {
                queue_.pop_back();
            }
        }

        // Returns the worst (largest est_dist) candidate.
        [[nodiscard]] const Candidate& worst() const { return queue_.back(); }

        [[nodiscard]] size_t size() const { return queue_.size(); }

        [[nodiscard]] const std::vector<Candidate>& candidates() const { return queue_; }

       private:
        size_t capacity_;
        // Sorted in ascending order by record.est_dist so that the worst is at the back.
        std::vector<Candidate> queue_;
    };

   private:
    friend maxheap<std::pair<float, PID>> detail::search_knn_avx2(
        HierarchicalNSW&, const float*, size_t
    );
    friend maxheap<std::pair<float, PID>> detail::search_knn_avx512_core(
        HierarchicalNSW&, const float*, size_t
    );
    friend maxheap<std::pair<float, PID>> detail::search_knn_avx512_popcnt(
        HierarchicalNSW&, const float*, size_t
    );

    static constexpr PID kMaxLabelOperationLock = 65536;
    // Bumped whenever the on-disk element layout changes; load() rejects any
    // other version outright, and there is no migration shim.
    //   v2 -> v3  the x+y layout dropped its duplicated copy of the base code
    //   v3 -> v4  the 2+x base region became two 1-bit planes around its
    //             factors (TwoBitBaseDataMap) instead of a packed 2-bit
    //             excode, and base_bits is gated to {1, 2}
    static constexpr uint32_t kFormatVersion = 4;
    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count_{0};  // current number of elements
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};
    size_t ef_construction_{0};
    size_t ef_{0};
    MetricType metric_type_;

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    // Locks operations with element by label value
    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global_;
    std::vector<std::mutex> link_list_locks_;

    PID enterpoint_node_{0};

    size_t size_links_level0_{0};

    size_t offsetBinData_{0}, offsetExData_{0}, label_offset_{0};
    size_t size_bin_data_{0}, size_ex_data_{0};
    size_t ex_bits_{0};

    // Two-level (2+x) storage/search path fields, set by the XyQuantBits
    // constructor when base_bits == 2. Unused (left zero) otherwise, which
    // keeps the fields above and the classic popcount/BinData/ExData path
    // fully in play.
    //
    // two_level_mode_ selects the storage path and, since base_bits is gated
    // to {1, 2} with 1 routed to the legacy layout, it now holds exactly when
    // base_bits_ == 2. The two are kept as separate fields anyway so the file
    // format does not have to be re-versioned if that gate ever widens.
    bool two_level_mode_{false};
    size_t base_bits_{1};
    size_t offsetBaseData_{0}, offsetExtraData_{0};
    size_t size_base_data_{0}, size_extra_data_{0};

    // Layout: (# of edges + edges) + (cluster_id) + (External_id) + (BinData) + (ExData)
    char* data_level0_memory_{nullptr};
    char** linkLists_{nullptr};
    std::vector<int> element_levels_;  // keeps level of each element

    size_t num_cluster_{0};
    size_t dim_{0};
    size_t padded_dim_{0};

    char* centroids_memory_{nullptr};

    mutable std::mutex label_lookup_lock_;  // lock for label_lookup_
    std::unordered_map<PID, PID> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations_{0};
    mutable std::atomic<long> metric_hops_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    float (*ip_func_)(const float*, const uint8_t*, size_t);

    Rotator<float>* rotator_ = nullptr;

    quant::RabitqConfig query_config_;

    struct EstimateRecord {
        float ip_x0_qr;
        float est_dist;
        float low_dist;

        bool operator<(const EstimateRecord& other) const {
            return this->est_dist < other.est_dist;
        }
    };

    float (*raw_dist_func_)(const float* __restrict__, const float* __restrict__, size_t);

    void free_memory() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (PID i = 0; i < cur_element_count_; i++) {
            if (element_levels_[i] > 0) {
                free(linkLists_[i]);
            }
        }
        free(reinterpret_cast<void*>(linkLists_));
        linkLists_ = nullptr;
        cur_element_count_ = 0;

        free(centroids_memory_);

        delete rotator_;
        rotator_ = nullptr;
    }

    void set_ef(size_t ef) { ef_ = ef; }

    std::mutex& get_lable_op_mutex(PID label) const {
        // calculate hash
        size_t lock_id = label & (kMaxLabelOperationLock - 1);
        return label_op_locks_[lock_id];
    }

    PID get_external_label(PID internal_id) const {
        PID return_label;
        memcpy(
            &return_label,
            (data_level0_memory_ + (internal_id * size_data_per_element_) + label_offset_),
            sizeof(PID)
        );
        return return_label;
    }

    void set_external_label(PID internal_id, PID label) const {
        memcpy(
            (data_level0_memory_ + (internal_id * size_data_per_element_) + label_offset_),
            &label,
            sizeof(PID)
        );
    }

    PID* get_external_label_pt(PID internal_id) const {
        return reinterpret_cast<PID*>(
            data_level0_memory_ + (internal_id * size_data_per_element_) + label_offset_
        );
    }

    char* get_bindata_by_internalid(PID internal_id) const {
        return reinterpret_cast<char*>(
            data_level0_memory_ + (internal_id * size_data_per_element_) + offsetBinData_
        );
    }

    char* get_exdata_by_internalid(PID internal_id) const {
        return reinterpret_cast<char*>(
            data_level0_memory_ + (internal_id * size_data_per_element_) + offsetExData_
        );
    }

    char* get_basedata_by_internalid(PID internal_id) const {
        return reinterpret_cast<char*>(
            data_level0_memory_ + (internal_id * size_data_per_element_) + offsetBaseData_
        );
    }

    char* get_extradata_by_internalid(PID internal_id) const {
        return reinterpret_cast<char*>(
            data_level0_memory_ + (internal_id * size_data_per_element_) + offsetExtraData_
        );
    }

    PID get_clusterid_by_internalid(PID internal_id) const {
        return *(reinterpret_cast<PID*>(
            data_level0_memory_ + (internal_id * size_data_per_element_) +
            size_links_level0_
        ));
    }

    char* get_clusterid_pt(PID internal_id) const {
        return reinterpret_cast<char*>(
            data_level0_memory_ + (internal_id * size_data_per_element_) +
            size_links_level0_
        );
    }

    int get_random_level(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return static_cast<int>(r);
    }

    size_t get_max_elements() const { return max_elements_; }

    size_t get_current_element_count() const { return cur_element_count_; }

    PID* get_linklist(PID internal_id, int level) const {
        return reinterpret_cast<PID*>(
            linkLists_[internal_id] + ((level - 1) * size_links_per_element_)
        );
    }

    PID* get_linklist0(PID internal_id) const {
        return reinterpret_cast<PID*>(
            data_level0_memory_ + (internal_id * size_data_per_element_)
        );
    }

    static unsigned short int get_list_count(const PID* ptr) {
        return *(reinterpret_cast<const unsigned short int*>(ptr));
    }

    static void set_list_count(PID* ptr, unsigned short int size) {
        *(reinterpret_cast<unsigned short int*>(ptr)) = size;
    }

    // ANN Search
    template <class Kernel>
    void
    get_bin_est_direct(std::vector<float>&, SplitSingleQuery<float>&, PID, HierarchicalNSW::EstimateRecord&);

    template <class Kernel>
    void
    get_full_est_direct(std::vector<float>&, SplitSingleQuery<float>&, PID, HierarchicalNSW::EstimateRecord&)
        const;

    // Progressive distance for the 2+x path: estimate at 1 bit, then 2, then
    // the full 2+x width, stopping as soon as low_dist alone settles whether
    // the candidate can enter the result set. See hnsw_two_level.hpp.
    // Returns false if a filter stage settled that this candidate cannot
    // reach `threshold`, true if it ran to the full width. That return value
    // -- not res.low_dist -- is the admission decision, because res.low_dist
    // ends up holding the *refined* bound once stage 3 runs.
    template <class Kernel>
    bool progressive_dist(
        std::vector<float>& q_to_centroids,
        TwoLevelQuery<float>& query,
        PID currObj,
        float threshold,
        HierarchicalNSW::EstimateRecord& res
    ) const;

    maxheap<std::pair<float, PID>> search_knn(const float*, size_t);

    // Two-level counterpart of search_knn_direct<Kernel>'s body; factored out
    // (see hnsw_two_level.hpp) so search_knn_direct itself stays a thin
    // dispatch.
    template <class Kernel>
    maxheap<std::pair<float, PID>> search_knn_direct_two_level(
        const float* rotated_query, size_t TOPK, std::vector<float>& q_to_centroids
    );

    template <class Kernel>
    maxheap<std::pair<float, PID>> search_knn_direct(const float*, size_t);

    template <class Kernel>
    void searchBaseLayerST_AdaptiveRerankOptDirect(
        PID ep_id,
        size_t ef,
        size_t TOPK,
        SplitSingleQuery<float>& query_wrapper,
        std::vector<float>& q_to_centroids,
        const float* query,
        BoundedKNN& boundedKNN
    );

    template <class Kernel>
    void searchBaseLayerST_TwoLevel(
        PID ep_id,
        size_t ef,
        size_t TOPK,
        TwoLevelQuery<float>& query,
        std::vector<float>& q_to_centroids,
        BoundedKNN& boundedKNN
    );

    // Construction
    // Currently only support index construction with non-quantized vectors
    float get_data_dist(PID obj1, PID obj2) {
        PID label1 = get_external_label(obj1);
        PID label2 = get_external_label(obj2);
        return raw_dist_func_(
            rawDataPtr_ + (label1 * dim_), rawDataPtr_ + (label2 * dim_), dim_
        );
    }

    // quantize_fn(cur_c, cluster_id, rotated_data) performs the actual
    // quantization + storage step; construct() builds the right closure for
    // the index's mode (1+y vs 2+x) so add_point's link/level bookkeeping
    // doesn't need to be duplicated per mode.
    void add_point(PID, PID, const std::function<void(PID, PID, const float*)>&);

    // Builds construct()'s quantize_fn closure for the 2+x mode (see
    // hnsw_two_level.hpp); the 1+y branch is unchanged.
    std::function<void(PID, PID, const float*)> make_two_level_quantize_fn(bool faster);

    maxheap<std::pair<float, PID>> search_base_layer(PID, PID, int);

    // Constructor building blocks, shared by the 1+y and 2+x entry points.
    void init_geometry_and_graph(size_t, size_t, size_t, size_t);
    void init_legacy_layout();
    void init_two_level_layout();
    void allocate_storage(size_t);

    PID mutually_connect_new_element(PID, maxheap<std::pair<float, PID>>&, int);

    void get_neighbors_by_heuristic2(maxheap<std::pair<float, PID>>&, size_t);
};

// Everything both constructors do before they know anything about the code
// width: geometry, the rotator, and the graph parameters. Leaves
// size_links_level0_ and label_offset_ set, so a layout initializer can lay
// its data regions out immediately after the label.
inline void HierarchicalNSW::init_geometry_and_graph(
    size_t max_elements, size_t dim, size_t M, size_t ef_construction
) {
    max_elements_ = max_elements;
    dim_ = dim;
    rotator_ = choose_rotator<float>(
        dim, RotatorType::FhtKacRotator, round_up_to_multiple(dim_, 64)
    );
    padded_dim_ = rotator_->size();
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);

    if (M <= 10000) {
        M_ = M;
    } else {
        std::cout << "warning: M parameter exceeds 10000 which may lead to adverse effects."
                  << '\n';
        std::cout << "Cap to 10000 will be applied for the rest of the processing." << '\n';
        M_ = 10000;
    }

    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    // (# of edges + edges) + (cluster_id) + (external label) + <data regions>
    size_links_level0_ = maxM0_ * sizeof(PID) + sizeof(PID);
    label_offset_ = size_links_level0_ + sizeof(PID);
}

// 1+y layout: BinData (1-bit code + 3 factors) then ExData. Reads ex_bits_.
inline void HierarchicalNSW::init_legacy_layout() {
    two_level_mode_ = false;
    base_bits_ = 1;
    ip_func_ = select_excode_ipfunc(ex_bits_);

    size_bin_data_ = BinDataMap<float>::data_bytes(padded_dim_);
    size_ex_data_ = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    offsetBinData_ = label_offset_ + sizeof(PID);
    offsetExData_ = offsetBinData_ + size_bin_data_;
    size_data_per_element_ = offsetExData_ + size_ex_data_;
}

// 2+x layout: the two-plane base region (TwoBitBaseDataMap) then ExData.
// Reads ex_bits_. There is no base-side excode ip function here -- the base
// code is two 1-bit planes, read by the binary kernels.
inline void HierarchicalNSW::init_two_level_layout() {
    two_level_mode_ = true;
    base_bits_ = 2;
    ip_func_ = (ex_bits_ > 0) ? select_excode_ipfunc(ex_bits_) : nullptr;

    size_base_data_ = TwoBitBaseDataMap<float>::data_bytes(padded_dim_);
    size_extra_data_ = ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
    offsetBaseData_ = label_offset_ + sizeof(PID);
    offsetExtraData_ = offsetBaseData_ + size_base_data_;
    size_data_per_element_ = offsetExtraData_ + size_extra_data_;
}

// Allocate level-0 storage and initialize the rest of the graph state. Runs
// after a layout initializer has set size_data_per_element_.
inline void HierarchicalNSW::allocate_storage(size_t random_seed) {
    data_level0_memory_ =
        memory::huge_page_allocate<char>(max_elements_ * size_data_per_element_);
    if (data_level0_memory_ == nullptr) {
        throw std::runtime_error("Not enough memory");
    }

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    cur_element_count_ = 0;

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

    // initializations for special treatment of the first node
    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = reinterpret_cast<char**>(malloc(sizeof(void*) * max_elements_));
    if (linkLists_ == nullptr) {
        throw std::runtime_error("Not enough memory: HNSW failed to allocate linklists");
    }
    size_links_per_element_ = maxM_ * sizeof(PID) + sizeof(PID);
    mult_ = 1 / log(1.0 * static_cast<double>(M_));
    revSize_ = 1.0 / mult_;

    // SplitSingleQuery and TwoLevelQuery both quantize the query to 4 bits,
    // so one config serves either path.
    this->query_config_ =
        quant::faster_config(padded_dim_, SplitSingleQuery<float>::kNumBits);
}

inline HierarchicalNSW::HierarchicalNSW(
    size_t max_elements,
    size_t dim,
    size_t total_bits,
    size_t M,
    size_t ef_construction,
    size_t random_seed,
    MetricType metric_type
)
    : metric_type_(metric_type)
    , label_op_locks_(kMaxLabelOperationLock)
    , link_list_locks_(max_elements)
    , element_levels_(max_elements)
    , raw_dist_func_((metric_type == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>) {
    init_geometry_and_graph(max_elements, dim, M, ef_construction);

    // 9, not kMaxCombinedBits (10): the 1+y layout spends total_bits - 1 on
    // the ex code, and pack_excode / select_excode_ipfunc only go to 8 bits.
    // A 10-bit code is reachable only as the two-level 2+8, where the base
    // is two 1-bit planes and the ex code stays within 8.
    constexpr size_t kMaxLegacyTotalBits = 9;
    if (total_bits < 1 || total_bits > kMaxLegacyTotalBits) {
        throw std::invalid_argument(
            "HierarchicalNSW: total_bits must be in [1, " +
            std::to_string(kMaxLegacyTotalBits) + "], got " + std::to_string(total_bits)
        );
    }
    ex_bits_ = total_bits - 1;

    init_legacy_layout();
    allocate_storage(random_seed);
}

inline HierarchicalNSW::~HierarchicalNSW() { free_memory(); }

inline void HierarchicalNSW::save(const char* filename) const {
    std::ofstream output(filename, std::ios::binary);

    output.write(reinterpret_cast<const char*>(&kFormatVersion), sizeof(kFormatVersion));

    output.write(reinterpret_cast<const char*>(&max_elements_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&cur_element_count_), sizeof(size_t));

    output.write(reinterpret_cast<const char*>(&dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&padded_dim_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&num_cluster_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&two_level_mode_), sizeof(bool));
    output.write(reinterpret_cast<const char*>(&base_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&ex_bits_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&metric_type_), sizeof(metric_type_));

    output.write(reinterpret_cast<const char*>(&size_bin_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&size_ex_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&size_base_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&size_extra_data_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&size_links_level0_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&offsetBinData_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&offsetExData_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&offsetBaseData_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&offsetExtraData_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&label_offset_), sizeof(PID));
    output.write(reinterpret_cast<const char*>(&size_data_per_element_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&size_links_per_element_), sizeof(size_t));

    output.write(reinterpret_cast<const char*>(&maxlevel_), sizeof(int));
    output.write(reinterpret_cast<const char*>(&enterpoint_node_), sizeof(PID));

    output.write(reinterpret_cast<const char*>(&M_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&maxM_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&maxM0_), sizeof(size_t));
    output.write(reinterpret_cast<const char*>(&mult_), sizeof(double));
    output.write(reinterpret_cast<const char*>(&ef_construction_), sizeof(size_t));

    std::cout << "cur_element_count = " << cur_element_count_ << '\n';

    output.write(
        reinterpret_cast<const char*>(centroids_memory_),
        static_cast<std::streamsize>(num_cluster_ * padded_dim_ * sizeof(float))
    );

    output.write(
        reinterpret_cast<const char*>(data_level0_memory_),
        static_cast<std::streamsize>(cur_element_count_ * size_data_per_element_)
    );

    for (size_t i = 0; i < cur_element_count_; i++) {
        unsigned int link_list_size =
            element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
        output.write(reinterpret_cast<const char*>(&link_list_size), sizeof(unsigned int));
        if (link_list_size != 0) {
            output.write(reinterpret_cast<const char*>(linkLists_[i]), link_list_size);
        }
    }

    rotator_->save(output);
    output.close();
}

inline void HierarchicalNSW::load(const char* filename) {
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open()) {
        throw std::runtime_error("Cannot open file");
    }

    free_memory();

    uint32_t format_version = 0;
    input.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    if (format_version != kFormatVersion) {
        throw std::runtime_error(
            "HierarchicalNSW::load: unsupported index file format version (got " +
            std::to_string(format_version) + ", expected " +
            std::to_string(kFormatVersion) +
            ") -- rebuild the index with the current library version"
        );
    }

    input.read(reinterpret_cast<char*>(&max_elements_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&cur_element_count_), sizeof(size_t));

    input.read(reinterpret_cast<char*>(&dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&padded_dim_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&num_cluster_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&two_level_mode_), sizeof(bool));
    input.read(reinterpret_cast<char*>(&base_bits_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&ex_bits_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&metric_type_), sizeof(metric_type_));
    raw_dist_func_ =
        (metric_type_ == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>;

    if (two_level_mode_ && base_bits_ != 2) {
        throw std::runtime_error(
            "HierarchicalNSW::load: two-level index with base_bits != 2; this build "
            "only supports base_bits 1 (legacy layout) and 2"
        );
    }
    // The 2+x base code is two 1-bit planes read by the binary kernels, so
    // only the extra region needs an excode ip function.
    ip_func_ =
        (!two_level_mode_ || ex_bits_ > 0) ? select_excode_ipfunc(ex_bits_) : nullptr;

    input.read(reinterpret_cast<char*>(&size_bin_data_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&size_ex_data_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&size_base_data_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&size_extra_data_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&size_links_level0_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&offsetBinData_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&offsetExData_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&offsetBaseData_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&offsetExtraData_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&label_offset_), sizeof(PID));
    input.read(reinterpret_cast<char*>(&size_data_per_element_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&size_links_per_element_), sizeof(size_t));

    input.read(reinterpret_cast<char*>(&maxlevel_), sizeof(int));
    input.read(reinterpret_cast<char*>(&enterpoint_node_), sizeof(PID));

    input.read(reinterpret_cast<char*>(&M_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&maxM_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&maxM0_), sizeof(size_t));
    input.read(reinterpret_cast<char*>(&mult_), sizeof(double));
    input.read(reinterpret_cast<char*>(&ef_construction_), sizeof(size_t));

    const size_t centroids_bytes = num_cluster_ * padded_dim_ * sizeof(float);
    centroids_memory_ = memory::huge_page_allocate<char>(centroids_bytes);
    if (centroids_memory_ == nullptr) {
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate centroids"
        );
    }

    input.read(
        centroids_memory_,
        static_cast<std::streamsize>(num_cluster_ * padded_dim_ * sizeof(float))
    );

    data_level0_memory_ =
        memory::huge_page_allocate<char>(max_elements_ * size_data_per_element_);

    input.read(
        data_level0_memory_,
        static_cast<std::streamsize>(cur_element_count_ * size_data_per_element_)
    );

    std::cout << "cur_element_count = " << cur_element_count_ << '\n';

    std::vector<std::mutex>(max_elements_).swap(link_list_locks_);
    std::vector<std::mutex>(kMaxLabelOperationLock).swap(label_op_locks_);

    linkLists_ = reinterpret_cast<char**>(malloc(sizeof(void*) * max_elements_));
    if (linkLists_ == nullptr) {
        throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists"
        );
    }

    element_levels_ = std::vector<int>(max_elements_);
    revSize_ = 1.0 / mult_;
    ef_ = 10;

    for (size_t i = 0; i < cur_element_count_; i++) {
        label_lookup_[get_external_label(i)] = i;
        unsigned int link_list_size;
        input.read(reinterpret_cast<char*>(&link_list_size), sizeof(unsigned int));
        if (link_list_size == 0) {
            element_levels_[i] = 0;
            linkLists_[i] = nullptr;
        } else {
            element_levels_[i] = static_cast<int>(link_list_size / size_links_per_element_);
            linkLists_[i] = reinterpret_cast<char*>(malloc(link_list_size));
            if (linkLists_[i] == nullptr) {
                throw std::runtime_error(
                    "Not enough memory: loadIndex failed to allocate linklist"
                );
            }
            input.read(linkLists_[i], link_list_size);
        }
    }

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

    rotator_ = choose_rotator<float>(
        dim_, RotatorType::FhtKacRotator, round_up_to_multiple(dim_, 64)
    );
    if (rotator_->size() != padded_dim_) {
        std::cerr << "Bad padded_dim_ for rotator in hnsw.load()\n";
        exit(1);
    }
    rotator_->load(input);
    input.close();

    this->query_config_ =
        quant::faster_config(padded_dim_, SplitSingleQuery<float>::kNumBits);
}

inline void HierarchicalNSW::construct(
    size_t cluster_num,
    const float* centroids,
    size_t data_num,
    const float* data,
    PID* cluster_ids,
    size_t num_threads = 0,
    bool faster = false
) {
    num_cluster_ = cluster_num;
    const size_t centroids_bytes = num_cluster_ * padded_dim_ * sizeof(float);
    centroids_memory_ = reinterpret_cast<char*>(malloc(centroids_bytes));
    if (centroids_memory_ == nullptr) {
        throw std::runtime_error("Not enough memory: HNSW failed to allocate centroids");
    }

    for (size_t i = 0; i < cluster_num; ++i) {
        this->rotator_->rotate(
            centroids + (i * dim_),
            reinterpret_cast<float*>(centroids_memory_) + (i * padded_dim_)
        );
    }

    std::function<void(PID, PID, const float*)> quantize_fn;
    if (!two_level_mode_) {
        quant::RabitqConfig config;
        if (faster) {
            config = quant::faster_config(padded_dim_, ex_bits_ + 1);
        }
        quantize_fn = [this, config](PID cur_c, PID cluster_id, const float* rotated_data) {
            quant::quantize_split_single(
                rotated_data,
                reinterpret_cast<float*>(centroids_memory_) + (cluster_id * padded_dim_),
                padded_dim_,
                ex_bits_,
                get_bindata_by_internalid(cur_c),
                get_exdata_by_internalid(cur_c),
                metric_type_,
                config
            );
        };
    } else {
        quantize_fn = make_two_level_quantize_fn(faster);
    }

    std::cout << "Start HierarchicalNSW construction..." << '\n';
    rawDataPtr_ = data;
    std::cout << "Build edges with non-quantized vectors..." << '\n';
    rabitqlib::ivf::parallel_for(
        0,
        data_num,
        num_threads,
        [&](size_t idx, size_t /*threadId*/) {
            add_point(idx, cluster_ids[idx], quantize_fn);
        }
    );
}

inline void HierarchicalNSW::add_point(
    PID label,
    PID cluster_id,
    const std::function<void(PID, PID, const float*)>& quantize_fn
) {
    std::unique_lock<std::mutex> lock_label(get_lable_op_mutex(label));

    int level = -1;
    PID cur_c = 0;
    {
        std::unique_lock<std::mutex> lock_table(label_lookup_lock_);
        if (label_lookup_.find(label) != label_lookup_.end()) {
            throw std::runtime_error(
                "Currently not support replacement of existing elements, only support "
                "inserting elements with distinct labels"
            );
        }

        if (cur_element_count_ >= max_elements_) {
            throw std::runtime_error("The number of elements exceeds the specified limit");
        }

        cur_c = cur_element_count_;
        cur_element_count_++;
        label_lookup_[label] = cur_c;
    }

    std::unique_lock<std::mutex> lock_el(link_list_locks_[cur_c]);
    int curlevel = get_random_level(mult_);
    if (level > 0) {
        curlevel = level;
    }

    element_levels_[cur_c] = curlevel;
    std::unique_lock<std::mutex> templock(global_);
    int maxlevelcopy = maxlevel_;
    if (curlevel <= maxlevelcopy) {
        templock.unlock();
    }
    PID curr_obj = enterpoint_node_;

    // initialize the current memory.
    memset(
        data_level0_memory_ + (cur_c * size_data_per_element_), 0, size_data_per_element_
    );

    // Initialisation of label and cluster id
    memcpy(get_external_label_pt(cur_c), &label, sizeof(PID));
    memcpy(get_clusterid_pt(cur_c), &cluster_id, sizeof(PID));

    // Quantize raw data and initialize quantized data
    std::vector<float> rotated_data(padded_dim_);
    rotator_->rotate(rawDataPtr_ + (label * dim_), rotated_data.data());
    quantize_fn(cur_c, cluster_id, rotated_data.data());

    // If the current vertex is at level >0, it needs some space to store the extra edges.
    if (curlevel > 0) {
        linkLists_[cur_c] =
            static_cast<char*>(malloc((size_links_per_element_ * curlevel) + 1));
        if (linkLists_[cur_c] == nullptr) {
            throw std::runtime_error(
                "Not enough memory: add_point failed to allocate linklist"
            );
        }
        memset(linkLists_[cur_c], 0, (size_links_per_element_ * curlevel) + 1);
    }

    if (static_cast<signed>(curr_obj) != -1) {
        if (curlevel < maxlevelcopy) {
            float curdist = get_data_dist(curr_obj, cur_c);
            for (int level = maxlevelcopy; level > curlevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int* data;
                    std::unique_lock<std::mutex> lock(link_list_locks_[curr_obj]);
                    data = get_linklist(curr_obj, level);
                    int size = get_list_count(data);

                    auto* datal = static_cast<PID*>(data + 1);
                    for (int i = 0; i < size; i++) {
                        PID cand = datal[i];
                        if (cand > max_elements_) {
                            throw std::runtime_error("cand error");
                        }
                        float d = get_data_dist(cand, cur_c);
                        if (d < curdist) {
                            curdist = d;
                            curr_obj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
            maxheap<std::pair<float, PID>> top_candidates =
                search_base_layer(curr_obj, cur_c, level);
            curr_obj = mutually_connect_new_element(cur_c, top_candidates, level);
        }
    } else {
        // Do nothing for the first element
        enterpoint_node_ = 0;
        maxlevel_ = curlevel;
    }

    // Releasing lock for the maximum level
    if (curlevel > maxlevelcopy) {
        enterpoint_node_ = cur_c;
        maxlevel_ = curlevel;
    }
}

inline maxheap<std::pair<float, PID>> HierarchicalNSW::search_base_layer(
    PID ep_id, PID cur_c, int layer
) {
    VisitedSet* vl = visited_list_pool_->get_free_vislist();

    maxheap<std::pair<float, PID>> top_candidates;
    minheap<std::pair<float, PID>> candidate_set;

    float lower_bound = get_data_dist(ep_id, cur_c);
    top_candidates.emplace(lower_bound, ep_id);
    candidate_set.emplace(lower_bound, ep_id);
    vl->set(ep_id);

    while (!candidate_set.empty()) {
        std::pair<float, PID> curr_el_pair = candidate_set.top();
        if (curr_el_pair.first > lower_bound && top_candidates.size() == ef_construction_) {
            break;
        }
        candidate_set.pop();

        PID cur_node_num = curr_el_pair.second;

        std::unique_lock<std::mutex> lock(link_list_locks_[cur_node_num]);

        int* data;
        if (layer == 0) {
            data = reinterpret_cast<int*>(get_linklist0(cur_node_num));
        } else {
            data = reinterpret_cast<int*>(get_linklist(cur_node_num, layer));
        }
        size_t size = get_list_count(reinterpret_cast<PID*>(data));
        auto* datal = reinterpret_cast<PID*>(data + 1);

        if (cur_node_num >= cur_element_count_ || size > maxM0_) {
            std::cerr << "BAD NODE: cur_node_num=" << cur_node_num
                      << " cur_element_count_=" << cur_element_count_.load()
                      << " size=" << size << " maxM0_=" << maxM0_ << " layer=" << layer
                      << " cur_c=" << cur_c << " ep_id=" << ep_id << "\n"
                      << std::flush;
            std::abort();
        }
        for (size_t dbg = 0; dbg < size; ++dbg) {
            if (datal[dbg] >= cur_element_count_) {
                std::cerr << "BAD NEIGHBOR: cur_node_num=" << cur_node_num << " idx=" << dbg
                          << " datal[idx]=" << datal[dbg]
                          << " cur_element_count_=" << cur_element_count_.load()
                          << " size=" << size << " layer=" << layer << " cur_c=" << cur_c
                          << " ep_id=" << ep_id << "\n"
                          << std::flush;
                std::abort();
            }
        }

        rabitqlib::memory::mem_prefetch_l1(
            reinterpret_cast<const char*>(
                rawDataPtr_ + (get_external_label(*datal) * dim_)
            ),
            padded_dim_ / 16
        );

        rabitqlib::memory::mem_prefetch_l1(
            reinterpret_cast<const char*>(
                rawDataPtr_ + (get_external_label(*(datal + 1)) * dim_)
            ),
            padded_dim_ / 16
        );

        for (size_t j = 0; j < size; j++) {
            PID candidate_id = *(datal + j);
            if (vl->get(candidate_id)) {
                continue;
            }
            vl->set(candidate_id);

            if (j < size - 1) {
                rabitqlib::memory::mem_prefetch_l1(
                    reinterpret_cast<const char*>(
                        rawDataPtr_ + (get_external_label(*(datal + j + 1)) * dim_)
                    ),
                    padded_dim_ / 16
                );
            }

            float dist1 = get_data_dist(candidate_id, cur_c);
            if (top_candidates.size() < ef_construction_ || lower_bound > dist1) {
                candidate_set.emplace(dist1, candidate_id);
                top_candidates.emplace(dist1, candidate_id);
                if (top_candidates.size() > ef_construction_) {
                    top_candidates.pop();
                }
                if (!top_candidates.empty()) {
                    lower_bound = top_candidates.top().first;
                }
            }
        }
    }
    visited_list_pool_->release_vis_list(vl);
    return top_candidates;
}

inline PID HierarchicalNSW::mutually_connect_new_element(
    PID cur_c, maxheap<std::pair<float, PID>>& top_candidates, int level
) {
    size_t max_m = level > 0 ? maxM_ : maxM0_;
    get_neighbors_by_heuristic2(top_candidates, M_);
    if (top_candidates.size() > M_) {
        throw std::runtime_error(
            "Should be not be more than M_ candidates returned by the heuristic"
        );
    }

    std::vector<PID> selected_neighbors;
    selected_neighbors.reserve(M_);
    while (top_candidates.size() > 0) {
        selected_neighbors.push_back(top_candidates.top().second);
        top_candidates.pop();
    }

    PID next_closest_entry_point = selected_neighbors.back();

    {
        PID* ll_cur;
        if (level == 0) {
            ll_cur = get_linklist0(cur_c);
        } else {
            ll_cur = get_linklist(cur_c, level);
        }

        if (*ll_cur > 0) {
            throw std::runtime_error(
                "The newly inserted element should have blank link list"
            );
        }

        set_list_count(ll_cur, selected_neighbors.size());
        auto* data = static_cast<PID*>(ll_cur + 1);
        for (size_t idx = 0; idx < selected_neighbors.size(); idx++) {
            if (data[idx] != 0) {
                throw std::runtime_error("Possible memory corruption");
            }
            if (level > element_levels_[selected_neighbors[idx]]) {
                throw std::runtime_error("Trying to make a link on a non-existent level");
            }

            data[idx] = selected_neighbors[idx];
        }
    }

    for (auto selected_neighbor : selected_neighbors) {
        std::unique_lock<std::mutex> lock(link_list_locks_[selected_neighbor]);

        PID* ll_other;
        if (level == 0) {
            ll_other = get_linklist0(selected_neighbor);
        } else {
            ll_other = get_linklist(selected_neighbor, level);
        }

        size_t sz_link_list_other = get_list_count(ll_other);

        if (sz_link_list_other > max_m) {
            throw std::runtime_error("Bad value of sz_link_list_other");
        }
        if (selected_neighbor == cur_c) {
            throw std::runtime_error("Trying to connect an element to itself");
        }
        if (level > element_levels_[selected_neighbor]) {
            throw std::runtime_error("Trying to make a link on a non-existent level");
        }

        auto* data = static_cast<PID*>(ll_other + 1);

        bool is_cur_c_present = false;
        for (size_t j = 0; j < sz_link_list_other; j++) {
            if (data[j] == cur_c) {
                is_cur_c_present = true;
                break;
            }
        }

        if (!is_cur_c_present) {
            if (sz_link_list_other < max_m) {
                data[sz_link_list_other] = cur_c;
                set_list_count(ll_other, sz_link_list_other + 1);
            } else {
                float d_max = get_data_dist(selected_neighbor, cur_c);
                maxheap<std::pair<float, PID>> candidates;
                candidates.emplace(d_max, cur_c);
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(get_data_dist(data[j], selected_neighbor), data[j]);
                }

                get_neighbors_by_heuristic2(candidates, max_m);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }

                set_list_count(ll_other, indx);
            }
        }
    }

    return next_closest_entry_point;
}

inline void HierarchicalNSW::get_neighbors_by_heuristic2(
    maxheap<std::pair<float, PID>>& top_candidates, size_t M
) {
    if (top_candidates.size() < M) {
        return;
    }

    minheap<std::pair<float, PID>> queue_closest;
    std::vector<std::pair<float, PID>> return_list;
    while (top_candidates.size() > 0) {
        queue_closest.emplace(top_candidates.top());
        top_candidates.pop();
    }

    while (queue_closest.size() > 0) {
        if (return_list.size() >= M) {
            break;
        }
        std::pair<float, PID> current_pair = queue_closest.top();
        float dist_to_query = current_pair.first;
        queue_closest.pop();
        bool good = true;

        for (std::pair<float, PID> second_pair : return_list) {
            float curdist = get_data_dist(second_pair.second, current_pair.second);
            if (curdist < dist_to_query) {
                good = false;
                break;
            }
        }
        if (good) {
            return_list.push_back(current_pair);
        }
    }

    for (std::pair<float, PID> current_pair : return_list) {
        top_candidates.emplace(current_pair);
    }
}

template <class Kernel>
inline void HierarchicalNSW::get_bin_est_direct(
    std::vector<float>& q_to_centroids,
    SplitSingleQuery<float>& query_wrapper,
    PID currObj,
    HierarchicalNSW::EstimateRecord& res
) {
    if (metric_type_ == METRIC_IP) {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        float error = q_to_centroids[get_clusterid_by_internalid(currObj) + num_cluster_];
        split_single_estdist_direct<Kernel>(
            get_bindata_by_internalid(currObj),
            query_wrapper,
            padded_dim_,
            res.ip_x0_qr,
            res.est_dist,
            res.low_dist,
            -norm,
            error
        );
    } else {
        // L2 distance
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        split_single_estdist_direct<Kernel>(
            get_bindata_by_internalid(currObj),
            query_wrapper,
            padded_dim_,
            res.ip_x0_qr,
            res.est_dist,
            res.low_dist,
            norm * norm,
            norm
        );
    }
}

template <class Kernel>
inline void HierarchicalNSW::get_full_est_direct(
    std::vector<float>& q_to_centroids,
    SplitSingleQuery<float>& query_wrapper,
    PID currObj,
    HierarchicalNSW::EstimateRecord& res
) const {
    if (metric_type_ == METRIC_IP) {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        float error = q_to_centroids[get_clusterid_by_internalid(currObj) + num_cluster_];
        split_single_fulldist_direct<Kernel>(
            get_bindata_by_internalid(currObj),
            get_exdata_by_internalid(currObj),
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            res.est_dist,
            res.low_dist,
            res.ip_x0_qr,
            -norm,
            error
        );
    } else {
        // L2 distance
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        split_single_fulldist_direct<Kernel>(
            get_bindata_by_internalid(currObj),
            get_exdata_by_internalid(currObj),
            ip_func_,
            query_wrapper,
            padded_dim_,
            ex_bits_,
            res.est_dist,
            res.low_dist,
            res.ip_x0_qr,
            norm * norm,
            norm
        );
    }
}

inline std::vector<std::vector<std::pair<float, PID>>> HierarchicalNSW::search(
    const float* queries, size_t query_num, size_t TOPK, size_t efSearch, size_t thread_num
) {
    set_ef(efSearch);
    std::vector<std::vector<std::pair<float, PID>>> results(query_num);
    rabitqlib::ivf::parallel_for(
        0,
        query_num,
        thread_num,
        [&](size_t idx, size_t /*threadId*/) {
            std::vector<float> rotated_query(padded_dim_);
            this->rotator_->rotate(queries + (idx * dim_), rotated_query.data());
            maxheap<std::pair<float, PID>> knn = search_knn(rotated_query.data(), TOPK);
            while (knn.size()) {
                results[idx].emplace_back(knn.top());
                knn.pop();
            }
            std::reverse(results[idx].begin(), results[idx].end());
        }
    );
    return results;
}

inline maxheap<std::pair<float, PID>> HierarchicalNSW::search_knn(
    const float* rotated_query, size_t TOPK
) {
    if (rabitqlib::cpu::has_avx512_popcnt()) {
        return detail::search_knn_avx512_popcnt(*this, rotated_query, TOPK);
    }
    if (rabitqlib::cpu::has_avx512_core() && rabitqlib::cpu::has_avx2()) {
        return detail::search_knn_avx512_core(*this, rotated_query, TOPK);
    }
    if (rabitqlib::cpu::has_avx2()) {
        return detail::search_knn_avx2(*this, rotated_query, TOPK);
    }

    throw std::runtime_error("HNSW search requires AVX2/FMA or AVX512 support");
}

template <class Kernel>
inline maxheap<std::pair<float, PID>> HierarchicalNSW::search_knn_direct(
    const float* rotated_query, size_t TOPK
) {
    maxheap<std::pair<float, PID>> result;
    if (cur_element_count_ == 0) {
        return result;
    }

    // Preprocess - get the distance from query to all centroids
    std::vector<float> q_to_centroids(num_cluster_);

    if (metric_type_ == METRIC_L2) {
        for (size_t i = 0; i < num_cluster_; i++) {
            q_to_centroids[i] = std::sqrt(raw_dist_func_(
                rotated_query,
                reinterpret_cast<float*>(centroids_memory_) + (i * padded_dim_),
                padded_dim_
            ));
        }
    } else if (metric_type_ == METRIC_IP) {
        q_to_centroids.resize(2 * num_cluster_);
        // first half as g_add, second half as g_error
        for (size_t i = 0; i < num_cluster_; i++) {
            q_to_centroids[i] = dot_product(
                rotated_query,
                reinterpret_cast<float*>(centroids_memory_) + (i * padded_dim_),
                padded_dim_
            );
            q_to_centroids[i + num_cluster_] = std::sqrt(euclidean_sqr(
                rotated_query,
                reinterpret_cast<float*>(centroids_memory_) + (i * padded_dim_),
                padded_dim_
            ));
        }
    }

    if (two_level_mode_) {
        return search_knn_direct_two_level<Kernel>(rotated_query, TOPK, q_to_centroids);
    }

    SplitSingleQuery<float> query_wrapper(
        rotated_query, padded_dim_, ex_bits_, query_config_, metric_type_
    );

    PID curr_obj = enterpoint_node_;
    EstimateRecord curest;

    get_bin_est_direct<Kernel>(q_to_centroids, query_wrapper, curr_obj, curest);

    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data;

            data = static_cast<unsigned int*>(get_linklist(curr_obj, level));
            int size = get_list_count(data);

            PID* datal = static_cast<PID*>(data + 1);
            for (int i = 0; i < size; i++) {
                PID cand = datal[i];
                if (cand > max_elements_) {
                    throw std::runtime_error("cand error");
                }

                EstimateRecord candest;
                get_bin_est_direct<Kernel>(q_to_centroids, query_wrapper, cand, candest);

                if (candest.est_dist < curest.est_dist) {
                    curest = candest;
                    curr_obj = cand;
                    changed = true;
                }
            }
        }
    }

    BoundedKNN boundedKnn(TOPK);
    searchBaseLayerST_AdaptiveRerankOptDirect<Kernel>(
        curr_obj,
        std::max(ef_, TOPK),
        TOPK,
        query_wrapper,
        q_to_centroids,
        rotated_query,
        boundedKnn
    );
    for (auto& candidate : boundedKnn.candidates()) {
        result.emplace(candidate.record.est_dist, get_external_label(candidate.id));
    }
    return result;
}

struct EstimateRecord {
    float est_dist;
    float low_dist;
};

template <class Kernel>
inline void HierarchicalNSW::searchBaseLayerST_AdaptiveRerankOptDirect(
    PID ep_id,
    size_t ef,
    size_t TOPK,
    SplitSingleQuery<float>& query_wrapper,
    std::vector<float>& q_to_centroids,
    [[maybe_unused]] const float* query,
    BoundedKNN& boundedKNN
) {
    VisitedSet* vl = visited_list_pool_->get_free_vislist();

    // Use our bounded priority queue instead of the maxheap.
    buffer::SearchBuffer<float> candidate_set(ef);

    float distk = 1e10;

    EstimateRecord start_estimate_record;
    if (ex_bits_ > 0) {
        get_full_est_direct<Kernel>(
            q_to_centroids, query_wrapper, ep_id, start_estimate_record
        );
    } else {
        get_bin_est_direct<Kernel>(
            q_to_centroids, query_wrapper, ep_id, start_estimate_record
        );
    }
    float est_dist = start_estimate_record.est_dist;
    float low_dist = start_estimate_record.low_dist;

    // Insert initial candidate.
    boundedKNN.insert({ResultRecord(est_dist, low_dist), ep_id});
    candidate_set.insert(ep_id, est_dist);

    distk = est_dist;

    vl->set(ep_id);

    const size_t prefetch_size = (((padded_dim_ / 8) + 63) / 64) + 1;
    const size_t prefetch_lookahead = 4;  // Number of neighbors to prefetch in advance.

    while (candidate_set.has_next()) {
        // Step 1 - get the next node to explore.
        PID current_node_id = candidate_set.pop();
        int* data = (int*)get_linklist0(current_node_id);
        size_t size = get_list_count((PID*)data);

        for (size_t p = 0; p < prefetch_lookahead; ++p) {
            rabitqlib::memory::mem_prefetch_l1(
                get_bindata_by_internalid(*(data + 1 + p)), prefetch_size
            );
        }
        // Iterate over neighbors. (List starts at index 1.)
        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);

            if (j + prefetch_lookahead <= size) {
                rabitqlib::memory::mem_prefetch_l1(
                    get_bindata_by_internalid(*(data + j + prefetch_lookahead)),
                    prefetch_size
                );
            }

            if (vl->get(candidate_id)) {
                continue;
            }
            vl->set(candidate_id);

            EstimateRecord candest;
            get_bin_est_direct<Kernel>(
                q_to_centroids, query_wrapper, candidate_id, candest
            );

            bool flag_update_KNNs = boundedKNN.size() < TOPK || candest.low_dist < distk;

            if (flag_update_KNNs) {
                // Compute the full estimate if promising.
                if (ex_bits_ > 0) {
                    get_full_est_direct<Kernel>(
                        q_to_centroids, query_wrapper, candidate_id, candest
                    );
                }
                Candidate cand{
                    ResultRecord(candest.est_dist, candest.low_dist),
                    static_cast<PID>(candidate_id)};
                boundedKNN.insert(cand);
                distk = boundedKNN.worst().record.est_dist;
            }

            if (!candidate_set.is_full(candest.est_dist)) {
                candidate_set.insert(candidate_id, candest.est_dist);
            }

            rabitqlib::memory::mem_prefetch_l2(
                (char*)get_linklist0(candidate_set.next_id()), 2
            );
        }
    }

    visited_list_pool_->release_vis_list(vl);
}

}  // namespace rabitqlib::hnsw

#include "rabitqlib/index/hnsw/hnsw_two_level.hpp"
