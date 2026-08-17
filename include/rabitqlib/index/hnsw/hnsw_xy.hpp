// Out-of-class definitions for HierarchicalNSW's x+y (XyQuantBits) mode --
// split out of hnsw.hpp purely to keep that file's size manageable. The
// class declaration (including these methods' signatures) still lives in
// hnsw.hpp, since C++ requires a class's member list to be contiguous;
// nothing here changes any pre-existing (base_bits_==1-only) behavior.
// Always included from the bottom of hnsw.hpp; the mutual #include here is
// guarded by hnsw.hpp's own #pragma once so this file also works if
// included directly.
#pragma once

#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace rabitqlib::hnsw {

// x+y constructor: always sets xy_mode_ = true, which activates the
// two-layer XY storage/search path -- including when base_bits == 1, which
// is a valid XyQuantBits value distinct from the *other* constructor's
// implicit base_bits_ == 1 (classic BinData/ExData/popcount path). The two
// must not be conflated: storage/search path selection always branches on
// xy_mode_, never on base_bits_'s value.
inline HierarchicalNSW::HierarchicalNSW(
    size_t max_elements,
    size_t dim,
    XyQuantBits bits,
    size_t M,
    size_t ef_construction,
    size_t random_seed,
    MetricType metric_type
)
    : metric_type_(metric_type)
    , label_op_locks_(kMaxLabelOperationLock)
    , link_list_locks_(max_elements)
    , element_levels_(max_elements)
    , raw_dist_func_(
          (metric_type == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>
      ) {
    max_elements_ = max_elements;
    dim_ = dim;
    rotator_ = choose_rotator<float>(
        dim, RotatorType::FhtKacRotator, round_up_to_multiple(dim_, 64)
    );
    padded_dim_ = rotator_->size();
    assert(padded_dim_ % 64 == 0);
    assert(padded_dim_ >= dim_);

    xy_mode_ = true;
    base_bits_ = bits.base_bits;
    ex_bits_ = bits.extra_bits;

    if (base_bits_ < 1 || base_bits_ > 8 || ex_bits_ > 8 ||
        base_bits_ + ex_bits_ > quant::rabitq_impl::xy_bits::kMaxCombinedBits) {
        std::cerr << "Invalid base_bits/extra_bits in HierarchicalNSW::HierarchicalNSW\n";
        std::cerr << "Expected: 1<=base_bits<=8, 0<=extra_bits<=8, base_bits+extra_bits<="
                  << quant::rabitq_impl::xy_bits::kMaxCombinedBits
                  << "  Input: base_bits=" << base_bits_ << " extra_bits=" << ex_bits_
                  << '\n';
        std::cerr.flush();
        exit(1);
    }

    xy_base_ip_func_ = select_excode_ipfunc(base_bits_);
    ip_func_ = (ex_bits_ > 0) ? select_excode_ipfunc(ex_bits_) : nullptr;

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

    size_xy_base_data_ = XYDataMap<float>::data_bytes(padded_dim_, base_bits_, 0);
    size_xy_extra_data_ = XYDataMap<float>::data_bytes(padded_dim_, base_bits_, ex_bits_);
    size_links_level0_ = maxM0_ * sizeof(PID) + sizeof(PID);
    label_offset_ = size_links_level0_ + sizeof(PID);
    offsetXYBaseData_ = label_offset_ + sizeof(PID);
    offsetXYExtraData_ = offsetXYBaseData_ + size_xy_base_data_;
    size_data_per_element_ = offsetXYExtraData_ + size_xy_extra_data_;

    data_level0_memory_ =
        reinterpret_cast<char*>(malloc(max_elements_ * size_data_per_element_));
    if (data_level0_memory_ == nullptr) {
        throw std::runtime_error("Not enough memory");
    }

    level_generator_.seed(random_seed);
    update_probability_generator_.seed(random_seed + 1);

    cur_element_count_ = 0;

    visited_list_pool_ = std::make_unique<VisitedListPool>(1, max_elements_);

    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = reinterpret_cast<char**>(malloc(sizeof(void*) * max_elements_));
    if (linkLists_ == nullptr) {
        throw std::runtime_error("Not enough memory: HNSW failed to allocate linklists");
    }
    size_links_per_element_ = maxM_ * sizeof(PID) + sizeof(PID);
    mult_ = 1 / log(1.0 * static_cast<double>(M_));
    revSize_ = 1.0 / mult_;
    // query_config_ is unused in XY mode -- XYQuery reads the float query
    // directly via generic SIMD dot kernels, no query-side scalar
    // quantization step like SplitSingleQuery's popcount path needs.
}

// Builds construct()'s quantize_fn closure for XY mode; called from
// construct()'s xy_mode_ branch in hnsw.hpp.
inline std::function<void(PID, PID, const float*)> HierarchicalNSW::make_xy_quantize_fn(
    bool faster
) {
    quant::RabitqConfig base_config;
    quant::RabitqConfig extra_config;
    if (faster) {
        base_config = quant::faster_config(padded_dim_, base_bits_);
        extra_config = quant::faster_config(padded_dim_, base_bits_ + ex_bits_);
    }
    return [this, base_config, extra_config](
               PID cur_c, PID cluster_id, const float* rotated_data
           ) {
        quant::quantize_xy_two_layer(
            rotated_data,
            reinterpret_cast<float*>(centroids_memory_) + (cluster_id * padded_dim_),
            padded_dim_,
            base_bits_,
            ex_bits_,
            get_xybasedata_by_internalid(cur_c),
            get_xyextradata_by_internalid(cur_c),
            metric_type_,
            base_config,
            extra_config
        );
    };
}

// Cheap filter (layer 1: base_bits_-only, extra_bits=0). Mirrors
// get_bin_est_direct's shape but via the generic xy_distance path.
inline void HierarchicalNSW::get_xybase_est(
    std::vector<float>& q_to_centroids,
    XYQuery<float>& query_base,
    PID currObj,
    HierarchicalNSW::EstimateRecord& res
) const {
    if (metric_type_ == METRIC_IP) {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        float error = q_to_centroids[get_clusterid_by_internalid(currObj) + num_cluster_];
        xy_distance(
            get_xybasedata_by_internalid(currObj),
            xy_base_ip_func_,
            nullptr,
            query_base,
            padded_dim_,
            base_bits_,
            0,
            res.est_dist,
            res.low_dist,
            -norm,
            error
        );
    } else {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        xy_distance(
            get_xybasedata_by_internalid(currObj),
            xy_base_ip_func_,
            nullptr,
            query_base,
            padded_dim_,
            base_bits_,
            0,
            res.est_dist,
            res.low_dist,
            norm * norm,
            norm
        );
    }
}

// Refinement (layer 2: base_bits_+ex_bits_). Mirrors get_full_est_direct.
inline void HierarchicalNSW::get_xyfull_est(
    std::vector<float>& q_to_centroids,
    XYQuery<float>& query_full,
    PID currObj,
    HierarchicalNSW::EstimateRecord& res
) const {
    if (metric_type_ == METRIC_IP) {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        float error = q_to_centroids[get_clusterid_by_internalid(currObj) + num_cluster_];
        xy_distance(
            get_xyextradata_by_internalid(currObj),
            xy_base_ip_func_,
            ip_func_,
            query_full,
            padded_dim_,
            base_bits_,
            ex_bits_,
            res.est_dist,
            res.low_dist,
            -norm,
            error
        );
    } else {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        xy_distance(
            get_xyextradata_by_internalid(currObj),
            xy_base_ip_func_,
            ip_func_,
            query_full,
            padded_dim_,
            base_bits_,
            ex_bits_,
            res.est_dist,
            res.low_dist,
            norm * norm,
            norm
        );
    }
}

// XY-mode counterpart of search_knn_direct<Kernel>'s body; called from
// search_knn_direct's xy_mode_ branch in hnsw.hpp. No Kernel template
// param -- select_excode_ipfunc's dispatch is already resolved into
// xy_base_ip_func_/ip_func_ at construction/load time.
inline maxheap<std::pair<float, PID>> HierarchicalNSW::search_knn_direct_xy(
    const float* rotated_query, size_t TOPK, std::vector<float>& q_to_centroids
) {
    maxheap<std::pair<float, PID>> result;

    XYQuery<float> query_base(rotated_query, padded_dim_, base_bits_, 0, metric_type_);
    XYQuery<float> query_full(rotated_query, padded_dim_, base_bits_, ex_bits_, metric_type_);

    PID curr_obj = enterpoint_node_;
    EstimateRecord curest;
    get_xybase_est(q_to_centroids, query_base, curr_obj, curest);

    for (int level = maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data = static_cast<unsigned int*>(get_linklist(curr_obj, level));
            int size = get_list_count(data);

            PID* datal = static_cast<PID*>(data + 1);
            for (int i = 0; i < size; i++) {
                PID cand = datal[i];
                if (cand > max_elements_) {
                    throw std::runtime_error("cand error");
                }

                EstimateRecord candest;
                get_xybase_est(q_to_centroids, query_base, cand, candest);

                if (candest.est_dist < curest.est_dist) {
                    curest = candest;
                    curr_obj = cand;
                    changed = true;
                }
            }
        }
    }

    BoundedKNN boundedKnn(TOPK);
    searchBaseLayerST_AdaptiveRerankOptDirectXY(
        curr_obj,
        std::max(ef_, TOPK),
        TOPK,
        query_base,
        query_full,
        q_to_centroids,
        boundedKnn
    );
    for (auto& candidate : boundedKnn.candidates()) {
        result.emplace(candidate.record.est_dist, get_external_label(candidate.id));
    }
    return result;
}

// Mirrors searchBaseLayerST_AdaptiveRerankOptDirect 1:1 for the base_bits_>1
// XY path: get_xybase_est (layer 1, cheap) / get_xyfull_est (layer 2,
// refine) in place of get_bin_est_direct<Kernel>/get_full_est_direct<Kernel>.
// No Kernel template param -- select_excode_ipfunc's dispatch is already
// resolved into xy_base_ip_func_/ip_func_ at construction/load time.
inline void HierarchicalNSW::searchBaseLayerST_AdaptiveRerankOptDirectXY(
    PID ep_id,
    size_t ef,
    size_t TOPK,
    XYQuery<float>& query_base,
    XYQuery<float>& query_full,
    std::vector<float>& q_to_centroids,
    BoundedKNN& boundedKNN
) {
    HashBasedBooleanSet* vl = visited_list_pool_->get_free_vislist();

    buffer::SearchBuffer<float> candidate_set(ef);

    float distk = 1e10;

    EstimateRecord start_estimate_record;
    get_xyfull_est(q_to_centroids, query_full, ep_id, start_estimate_record);
    float est_dist = start_estimate_record.est_dist;
    float low_dist = start_estimate_record.low_dist;

    boundedKNN.insert({ResultRecord(est_dist, low_dist), ep_id});
    candidate_set.insert(ep_id, est_dist);

    distk = est_dist;

    vl->set(ep_id);

    const size_t prefetch_size = (((padded_dim_ / 8) + 63) / 64) + 1;
    const size_t prefetch_lookahead = 4;  // Number of neighbors to prefetch in advance.

    while (candidate_set.has_next()) {
        PID current_node_id = candidate_set.pop();
        int* data = (int*)get_linklist0(current_node_id);
        size_t size = get_list_count((PID*)data);

        for (size_t p = 0; p < prefetch_lookahead; ++p) {
            rabitqlib::memory::mem_prefetch_l1(
                get_xybasedata_by_internalid(*(data + 1 + p)), prefetch_size
            );
        }
        // Iterate over neighbors. (List starts at index 1.)
        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);

            if (j + prefetch_lookahead <= size) {
                rabitqlib::memory::mem_prefetch_l1(
                    get_xybasedata_by_internalid(*(data + j + prefetch_lookahead)),
                    prefetch_size
                );
            }

            if (vl->get(candidate_id)) {
                continue;
            }
            vl->set(candidate_id);

            EstimateRecord candest;
            get_xybase_est(q_to_centroids, query_base, candidate_id, candest);

            bool flag_update_KNNs = boundedKNN.size() < TOPK || candest.low_dist < distk;

            if (flag_update_KNNs) {
                // Compute the full estimate if promising.
                if (ex_bits_ > 0) {
                    get_xyfull_est(q_to_centroids, query_full, candidate_id, candest);
                }
                Candidate cand{
                    ResultRecord(candest.est_dist, candest.low_dist),
                    static_cast<PID>(candidate_id)
                };
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
