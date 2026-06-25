#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "rabitqlib/index/hnsw/hnsw.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::hnsw::HierarchicalNSW;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

int main(int argc, char* argv[]) {
    if (argc < 8) {
        std::cerr << "Usage: " << argv[0]
                  << " <arg1> <arg2> <arg3> <arg4> <arg5> <arg6> <arg7> <arg8>\n"
                  << "arg1: path for data file, format .fvecs\n"
                  << "arg2: path for centroids file, format .fvecs\n"
                  << "arg3: path for cluster_ids file, format .ivecs\n"
                  << "arg4: m (degree bound) for hnsw\n"
                  << "arg5: ef for indexing \n"
                  << "arg6: total number of bits for quantization\n"
                  << "arg7: path for saving index\n"
                  << "arg8: metric type (\"l2\" or \"ip\"), l2 by default\n"
                  << "arg9: if use faster quantization (\"true\" or \"false\"), false by "
                     "default\n";
        exit(1);
    }

    char* data_file = argv[1];
    char* centroid_file = argv[2];
    char* cid_file = argv[3];
    size_t m = atoi(argv[4]);
    size_t ef = atoi(argv[5]);
    size_t total_bits = atoi(argv[6]);
    char* index_file = argv[7];

    rabitqlib::MetricType metric_type = rabitqlib::METRIC_L2;
    if (argc > 8) {
        std::string metric_str(argv[8]);
        if (metric_str == "ip" || metric_str == "IP") {
            metric_type = rabitqlib::METRIC_IP;
        }
    }
    if (metric_type == rabitqlib::METRIC_IP) {
        std::cout << "Metric Type: IP\n";
    } else if (metric_type == rabitqlib::METRIC_L2) {
        std::cout << "Metric Type: L2\n";
    }

    bool faster_quant = false;
    if (argc > 9) {
        std::string faster_str(argv[9]);
        if (faster_str == "true") {
            faster_quant = true;
            std::cout << "Using faster quantize for indexing...\n";
        }
    }

    data_type data;
    data_type centroids;
    gt_type cluster_id;

    rabitqlib::load_vecs<float, data_type>(data_file, data);
    rabitqlib::load_vecs<float, data_type>(centroid_file, centroids);
    rabitqlib::load_vecs<uint32_t, gt_type>(cid_file, cluster_id);

    size_t num_points = data.rows();
    size_t dim = data.cols();

    size_t random_seed = 100;  // by default 100
    auto* hnsw = new rabitqlib::hnsw::HierarchicalNSW(
        num_points, dim, total_bits, m, ef, random_seed, metric_type
    );

    // Build the index with the first 80% of the data, then stream the
    // remaining 20% in via add() in 10%-of-dataset batches to exercise the
    // dynamic-add path.
    size_t build_count = (num_points * 80) / 100;
    size_t batch_size = num_points / 10;  // 10% of the dataset per batch
    if (batch_size == 0) {
        batch_size = 1;
    }

    rabitqlib::StopW stopw;
    stopw.reset();

    hnsw->construct(
        centroids.rows(),
        centroids.data(),
        build_count,
        data.data(),
        cluster_id.data(),
        0,
        faster_quant
    );

    float build_time = stopw.get_elapsed_micro() / 1e6;
    std::cout << "built index with " << build_count << " / " << num_points
              << " points in " << build_time << "s" << '\n';

    // Labels mirror the global row index so they stay consistent with the
    // built portion (construct uses the row index as the label).
    std::vector<PID> labels(num_points);
    for (size_t i = 0; i < num_points; i++) {
        labels[i] = static_cast<PID>(i);
    }

    stopw.reset();
    for (size_t start = build_count; start < num_points; start += batch_size) {
        size_t end = std::min(start + batch_size, num_points);
        // num_threads = 0 -> use all available cores for the batch insert.
        hnsw->add(data.data() + (start * dim), labels.data() + start, end - start, 0);
        std::cout << "added points [" << start << ", " << end << ")\n";
    }
    float add_time = stopw.get_elapsed_micro() / 1e6;

    std::cout << "added " << (num_points - build_count) << " points in " << add_time
              << "s" << '\n';
    std::cout << "total indexing time = " << (build_time + add_time) << "s" << '\n';
    hnsw->save(index_file);

    std::cout << "index saved..." << '\n';

    return 0;
}