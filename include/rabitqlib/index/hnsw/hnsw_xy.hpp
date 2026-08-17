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

    // base region: base_bits_ of code + 3 factors; extra region: ex_bits_ of
    // code + 2 factors (zero-sized at ex_bits_ == 0). base_bits_ + ex_bits_
    // bits per dimension in total -- the base code lives in exactly one place.
    size_xy_base_data_ = XyBaseDataMap<float>::data_bytes(padded_dim_, base_bits_);
    size_xy_extra_data_ = XyExtraDataMap<float>::data_bytes(padded_dim_, ex_bits_);
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
    // One config, for the one code that gets built: the combined
    // base_bits_ + ex_bits_ width. Both regions come out of that single
    // quantization.
    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, base_bits_ + ex_bits_);
    }
    return [this, config](PID cur_c, PID cluster_id, const float* rotated_data) {
        quant::quantize_xy_single(
            rotated_data,
            reinterpret_cast<float*>(centroids_memory_) + (cluster_id * padded_dim_),
            padded_dim_,
            base_bits_,
            ex_bits_,
            get_xybasedata_by_internalid(cur_c),
            get_xyextradata_by_internalid(cur_c),
            metric_type_,
            config
        );
    };
}

// Cheap filter (base_bits_ code only). Mirrors get_bin_est_direct's shape,
// generic SIMD dot in place of popcount. Leaves the base inner product in
// res.ip_x0_qr for get_xyfull_est to boost -- exactly how the 1+y path
// carries ip_x0_qr from the binary stage into split_distance_boosting.
inline void HierarchicalNSW::get_xybase_est(
    std::vector<float>& q_to_centroids,
    XYQuery<float>& query,
    PID currObj,
    HierarchicalNSW::EstimateRecord& res
) const {
    if (metric_type_ == METRIC_IP) {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        float error = q_to_centroids[get_clusterid_by_internalid(currObj) + num_cluster_];
        xy_base_estdist(
            get_xybasedata_by_internalid(currObj),
            xy_base_ip_func_,
            query,
            padded_dim_,
            base_bits_,
            res.ip_x0_qr,
            res.est_dist,
            res.low_dist,
            -norm,
            error
        );
    } else {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        xy_base_estdist(
            get_xybasedata_by_internalid(currObj),
            xy_base_ip_func_,
            query,
            padded_dim_,
            base_bits_,
            res.ip_x0_qr,
            res.est_dist,
            res.low_dist,
            norm * norm,
            norm
        );
    }
}

// Refinement to base_bits_+ex_bits_, via boosting: touches only the ex_bits_
// code, since res.ip_x0_qr already holds this vector's base inner product
// from get_xybase_est (every call site runs the filter stage first).
//
// Only res.est_dist is updated. res.low_dist keeps the filter stage's bound,
// which is the one the search loop prunes on -- BoundedKNN orders and drops
// candidates purely by est_dist, so a refined low_dist would be stored and
// never read. This mirrors split_distance_boosting, which likewise returns
// an estimate and no bound.
inline void HierarchicalNSW::get_xyfull_est(
    std::vector<float>& q_to_centroids,
    XYQuery<float>& query,
    PID currObj,
    HierarchicalNSW::EstimateRecord& res
) const {
    float g_add = 0;
    if (metric_type_ == METRIC_IP) {
        g_add = -q_to_centroids[get_clusterid_by_internalid(currObj)];
    } else {
        float norm = q_to_centroids[get_clusterid_by_internalid(currObj)];
        g_add = norm * norm;
    }

    res.est_dist = xy_distance_boosting(
        get_xyextradata_by_internalid(currObj),
        ip_func_,
        query,
        padded_dim_,
        ex_bits_,
        res.ip_x0_qr,
        g_add
    );
}

// XY-mode counterpart of search_knn_direct<Kernel>'s body; called from
// search_knn_direct's xy_mode_ branch in hnsw.hpp. No Kernel template
// param -- select_excode_ipfunc's dispatch is already resolved into
// xy_base_ip_func_/ip_func_ at construction/load time.
inline maxheap<std::pair<float, PID>> HierarchicalNSW::search_knn_direct_xy(
    const float* rotated_query, size_t TOPK, std::vector<float>& q_to_centroids
) {
    maxheap<std::pair<float, PID>> result;

    // A single query object: it carries both offset-binary corrections, one
    // per code width, so the filter and refine stages share it.
    XYQuery<float> query(rotated_query, padded_dim_, base_bits_, ex_bits_, metric_type_);

    PID curr_obj = enterpoint_node_;
    EstimateRecord curest;
    get_xybase_est(q_to_centroids, query, curr_obj, curest);

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
                get_xybase_est(q_to_centroids, query, cand, candest);

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
        curr_obj, std::max(ef_, TOPK), TOPK, query, q_to_centroids, boundedKnn
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
    XYQuery<float>& query,
    std::vector<float>& q_to_centroids,
    BoundedKNN& boundedKNN
) {
    HashBasedBooleanSet* vl = visited_list_pool_->get_free_vislist();

    buffer::SearchBuffer<float> candidate_set(ef);

    float distk = 1e10;

    EstimateRecord start_estimate_record;
    // Filter stage first even for the entry point: get_xyfull_est boosts the
    // base inner product this call leaves in ip_x0_qr.
    get_xybase_est(q_to_centroids, query, ep_id, start_estimate_record);
    if (ex_bits_ > 0) {
        get_xyfull_est(q_to_centroids, query, ep_id, start_estimate_record);
    }
    float est_dist = start_estimate_record.est_dist;
    float low_dist = start_estimate_record.low_dist;

    boundedKNN.insert({ResultRecord(est_dist, low_dist), ep_id});
    candidate_set.insert(ep_id, est_dist);

    distk = est_dist;

    vl->set(ep_id);

    const size_t prefetch_size = ((size_xy_base_data_ + 63) / 64) + 1;
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
            get_xybase_est(q_to_centroids, query, candidate_id, candest);

            bool flag_update_KNNs = boundedKNN.size() < TOPK || candest.low_dist < distk;

            if (flag_update_KNNs) {
                // Boost to the full estimate if promising -- reuses
                // candest.ip_x0_qr, so only the ex_bits_ code is touched.
                if (ex_bits_ > 0) {
                    get_xyfull_est(q_to_centroids, query, candidate_id, candest);
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
