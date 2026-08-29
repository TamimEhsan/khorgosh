// Out-of-class definitions for HierarchicalNSW's two-level (2+x) mode --
// split out of hnsw.hpp purely to keep that file's size manageable. The
// class declaration (including these methods' signatures) still lives in
// hnsw.hpp, since C++ requires a class's member list to be contiguous;
// nothing here changes any 1+y behavior.
// Always included from the bottom of hnsw.hpp; the mutual #include here is
// guarded by hnsw.hpp's own #pragma once so this file also works if
// included directly.
#pragma once

#include <limits>

#include "rabitqlib/index/hnsw/hnsw.hpp"

namespace rabitqlib::hnsw {

// XyQuantBits constructor. base_bits is gated to {1, 2}:
//
//   base_bits == 1  ->  the legacy 1+y layout and search path, unchanged.
//                       There is no separate 1-bit implementation to keep in
//                       sync, and an index built this way is byte-identical
//                       to one built through the (total_bits) constructor.
//   base_bits == 2  ->  the two-level layout: a base region holding the
//                       2-bit code as two 1-bit planes, plus the usual extra
//                       region, searched by progressive_dist.
//
// Wider bases are rejected here rather than removed from the library --
// BaseDataMap, xy_base_estdist and quantize_xy_single still support 1..8 and
// are still tested, so widening this gate later is a change to this check.
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
    , raw_dist_func_((metric_type == METRIC_IP) ? dot_product_dis<float> : euclidean_sqr<float>) {
    if (bits.base_bits != 1 && bits.base_bits != 2) {
        throw std::invalid_argument(
            "HierarchicalNSW: base_bits must be 1 or 2, got " +
            std::to_string(bits.base_bits)
        );
    }
    if (bits.extra_bits > 8) {
        throw std::invalid_argument(
            "HierarchicalNSW: extra_bits must be in [0, 8], got " +
            std::to_string(bits.extra_bits)
        );
    }
    if (bits.base_bits + bits.extra_bits > quant::rabitq_impl::xy_bits::kMaxCombinedBits) {
        throw std::invalid_argument(
            "HierarchicalNSW: base_bits + extra_bits must be <= " +
            std::to_string(quant::rabitq_impl::xy_bits::kMaxCombinedBits)
        );
    }

    init_geometry_and_graph(max_elements, dim, M, ef_construction);
    ex_bits_ = bits.extra_bits;

    if (bits.base_bits == 1) {
        init_legacy_layout();
    } else {
        init_two_level_layout();
    }

    allocate_storage(random_seed);
}

// Builds construct()'s quantize_fn closure for the 2+x mode; called from
// construct()'s two_level_mode_ branch in hnsw.hpp.
inline std::function<void(PID, PID, const float*)>
HierarchicalNSW::make_two_level_quantize_fn(bool faster) {
    // One config, for the one code that gets built: the combined
    // 2 + ex_bits_ width. Both regions come out of that single quantization.
    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, base_bits_ + ex_bits_);
    }
    return [this, config](PID cur_c, PID cluster_id, const float* rotated_data) {
        quant::quantize_two_level_single(
            rotated_data,
            reinterpret_cast<float*>(centroids_memory_) + (cluster_id * padded_dim_),
            padded_dim_,
            ex_bits_,
            get_basedata_by_internalid(cur_c),
            (ex_bits_ > 0) ? get_extradata_by_internalid(cur_c) : nullptr,
            metric_type_,
            config
        );
    };
}

// Progressive distance: spend bits only until the answer is settled.
//
// `threshold` is the distance a candidate must beat to be worth refining --
// the search loop passes the current k-th best estimate, or +infinity while
// the result set is underfull. Each stage produces a lower bound on this
// candidate's distance; the moment that bound reaches the threshold, no
// amount of further refinement can bring the candidate under it, so the
// remaining stages are skipped.
//
//   stage 1  popcount against the base code's high plane and the *quantized*
//            query. Touches only the block's leading padded_dim/8 + 12 bytes.
//   stage 2  two masked-load passes over both planes, *float* query. Exact
//            for the 2-bit code. Leaves ip_base in res.ip_x0_qr.
//   stage 3  boost to the full 2+x width using res.ip_x0_qr, reading only
//            the extra region, and refine low_dist to match.
//
// res.est_dist / res.low_dist hold whichever stage ran last, so a candidate
// that exits early contributes a coarser estimate to the traversal order
// than one that runs to stage 3. That is the intended trade.
//
// The return value, not res.low_dist, says whether the candidate may enter
// the result set: once stage 3 has run, res.low_dist is the refined bound,
// and admitting on that rejects candidates the filter stage had already
// let through (section 7.2 of docs/two-level-quant-plan.md).
template <class Kernel>
inline bool HierarchicalNSW::progressive_dist(
    std::vector<float>& q_to_centroids,
    TwoLevelQuery<float>& query,
    PID currObj,
    float threshold,
    HierarchicalNSW::EstimateRecord& res
) const {
    const PID cluster_id = get_clusterid_by_internalid(currObj);
    float g_add = 0;
    float g_error = 0;
    if (metric_type_ == METRIC_IP) {
        g_add = -q_to_centroids[cluster_id];
        g_error = q_to_centroids[cluster_id + num_cluster_];
    } else {
        const float norm = q_to_centroids[cluster_id];
        g_add = norm * norm;
        g_error = norm;
    }

    const char* base_data = get_basedata_by_internalid(currObj);

    two_level_estdist_1bit<Kernel>(
        base_data, query, padded_dim_, res.est_dist, res.low_dist, g_add, g_error
    );
    if (res.low_dist >= threshold) {
        return false;
    }

    two_level_estdist_2bit<Kernel>(
        base_data,
        query,
        padded_dim_,
        res.ip_x0_qr,
        res.est_dist,
        res.low_dist,
        g_add,
        g_error
    );
    if (res.low_dist >= threshold) {
        return false;
    }
    if (ex_bits_ == 0) {
        // No refine layer: stage 2 is the full width, and it admitted this
        // candidate.
        return true;
    }

    ConstTwoBitBaseDataMap<float> cur_base(base_data, padded_dim_);
    two_level_boosting(
        get_extradata_by_internalid(currObj),
        ip_func_,
        query,
        padded_dim_,
        ex_bits_,
        res.ip_x0_qr,
        cur_base.f_error(),
        res.est_dist,
        res.low_dist,
        g_add,
        g_error
    );
    return true;
}

// Two-level counterpart of search_knn_direct<Kernel>'s body; called from
// search_knn_direct's two_level_mode_ branch in hnsw.hpp.
template <class Kernel>
inline maxheap<std::pair<float, PID>> HierarchicalNSW::search_knn_direct_two_level(
    const float* rotated_query, size_t TOPK, std::vector<float>& q_to_centroids
) {
    constexpr float kNoThreshold = std::numeric_limits<float>::max();
    maxheap<std::pair<float, PID>> result;

    TwoLevelQuery<float> query(
        rotated_query, padded_dim_, ex_bits_, query_config_, metric_type_
    );

    PID curr_obj = enterpoint_node_;
    EstimateRecord curest;
    progressive_dist<Kernel>(q_to_centroids, query, curr_obj, kNoThreshold, curest);

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

                // The greedy step only cares whether this candidate beats the
                // current one, so curest.est_dist is exactly the threshold
                // that makes further refinement pointless.
                EstimateRecord candest;
                progressive_dist<Kernel>(
                    q_to_centroids, query, cand, curest.est_dist, candest
                );

                if (candest.est_dist < curest.est_dist) {
                    curest = candest;
                    curr_obj = cand;
                    changed = true;
                }
            }
        }
    }

    BoundedKNN boundedKnn(TOPK);
    searchBaseLayerST_TwoLevel<Kernel>(
        curr_obj, std::max(ef_, TOPK), TOPK, query, q_to_centroids, boundedKnn
    );
    for (auto& candidate : boundedKnn.candidates()) {
        result.emplace(candidate.record.est_dist, get_external_label(candidate.id));
    }
    return result;
}

// Mirrors searchBaseLayerST_AdaptiveRerankOptDirect 1:1, with the
// bin-estimate / full-estimate pair replaced by a single progressive_dist
// call that decides internally how far to go.
template <class Kernel>
inline void HierarchicalNSW::searchBaseLayerST_TwoLevel(
    PID ep_id,
    size_t ef,
    size_t TOPK,
    TwoLevelQuery<float>& query,
    std::vector<float>& q_to_centroids,
    BoundedKNN& boundedKNN
) {
    constexpr float kNoThreshold = std::numeric_limits<float>::max();

    VisitedSet* vl = visited_list_pool_->get_free_vislist();

    buffer::SearchBuffer<float> candidate_set(ef);

    float distk = 1e10;

    EstimateRecord start_estimate_record;
    progressive_dist<Kernel>(
        q_to_centroids, query, ep_id, kNoThreshold, start_estimate_record
    );
    float est_dist = start_estimate_record.est_dist;
    float low_dist = start_estimate_record.low_dist;

    boundedKNN.insert({ResultRecord(est_dist, low_dist), ep_id});
    candidate_set.insert(ep_id, est_dist);

    distk = est_dist;

    vl->set(ep_id);

    // Stage 1 reads only the base block's leading padded_dim/8 bytes of
    // high plane plus its three factors; plane_lo sits past that and is
    // pulled in on demand by stage 2.
    const size_t prefetch_size = (((padded_dim_ / 8) + 12 + 63) / 64) + 1;
    const size_t prefetch_lookahead = 4;  // Number of neighbors to prefetch in advance.

    while (candidate_set.has_next()) {
        PID current_node_id = candidate_set.pop();
        int* data = (int*)get_linklist0(current_node_id);
        size_t size = get_list_count((PID*)data);

        for (size_t p = 0; p < prefetch_lookahead; ++p) {
            rabitqlib::memory::mem_prefetch_l1(
                get_basedata_by_internalid(*(data + 1 + p)), prefetch_size
            );
        }
        // Iterate over neighbors. (List starts at index 1.)
        for (size_t j = 1; j <= size; j++) {
            int candidate_id = *(data + j);

            if (j + prefetch_lookahead <= size) {
                rabitqlib::memory::mem_prefetch_l1(
                    get_basedata_by_internalid(*(data + j + prefetch_lookahead)),
                    prefetch_size
                );
            }

            if (vl->get(candidate_id)) {
                continue;
            }
            vl->set(candidate_id);

            // While the result set is underfull nothing may be pruned, so no
            // threshold is passed and every stage runs. Once it is full, a
            // candidate is only worth refining if it could still displace the
            // current worst -- which is precisely low_dist < distk, the test
            // applied below.
            const float threshold = (boundedKNN.size() < TOPK) ? kNoThreshold : distk;

            EstimateRecord candest;
            // The return value already encodes
            // "boundedKNN.size() < TOPK || filter-stage low_dist < distk":
            // with threshold == +inf the stages never prune, and otherwise
            // they prune on exactly that comparison. Re-testing
            // candest.low_dist here would use stage 3's refined bound and
            // reject candidates the filter stage admitted.
            bool flag_update_KNNs = progressive_dist<Kernel>(
                q_to_centroids, query, candidate_id, threshold, candest
            );

            if (flag_update_KNNs) {
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
