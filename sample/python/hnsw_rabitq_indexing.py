import argparse
from time import time

from rabitqlib import HnswIndex

from utils import cluster_data, read_fvecs

# ──────────────────────────────────────────────
# Default configuration
# ──────────────────────────────────────────────
NUM_CLUSTERS = 256  # number of clusters
M = 16  # degree bound for HNSW
EF_CONSTRUCTION = 200  # ef for indexing
TOTAL_BITS = 8  # total number of bits for quantization
METRIC = "l2"  # "l2" or "ip"
FASTER_QUANT = False  # use faster quantization
NUM_THREADS = 16  # number of threads for build
# ──────────────────────────────────────────────


def main(args=None) -> None:
    # 1. Load data
    data = read_fvecs(args.data_file)
    print(f"Data shape: {data.shape}")

    # 2. Cluster with FAISS
    centroids, cluster_ids = cluster_data(
        data, args.num_clusters, args.metric, args.num_threads
    )
    print(f"Centroids: {centroids.shape}, cluster_ids: {cluster_ids.shape}")

    # 3. Build HNSW index
    n, dim = data.shape
    if args.base_bits is None:
        print(f"\nBuilding HNSW index: n={n}, dim={dim}, M={args.degree}, "
              f"ef={args.ef_construction}, bits={args.total_bits}, metric={args.metric}")
    else:
        print(f"\nBuilding HNSW index: n={n}, dim={dim}, M={args.degree}, "
              f"ef={args.ef_construction}, base_bits={args.base_bits}, "
              f"extra_bits={args.extra_bits or 0}, metric={args.metric}")

    idx = HnswIndex(
        dim=dim,
        max_elements=n,
        M=args.degree,
        ef_construction=args.ef_construction,
        nbits=args.total_bits,
        metric=args.metric,
        base_bits=args.base_bits,
        extra_bits=args.extra_bits,
    )

    t0 = time()
    idx.build(
        data,
        centroids,
        cluster_ids,
        num_threads=args.num_threads,
        fast_quantization=args.faster_quant,
    )
    print(f"Indexing time: {time() - t0:.2f}s")

    idx.save(args.index_file)
    print(f"Index saved → {args.index_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RaBitQ HNSW Manager")

    parser.add_argument("data_file", type=str, help="Path to the data file")
    parser.add_argument("index_file", type=str, help="Path to save the index")
    parser.add_argument(
        "--num-clusters",
        dest="num_clusters",
        type=int,
        metavar="INT",
        default=256,
        help="Number of clusters for quantization",
    )
    parser.add_argument(
        "--degree",
        dest="degree",
        type=int,
        metavar="INT",
        default=M,
        help="Degree bound for HNSW",
    )
    parser.add_argument(
        "--ef-construction",
        dest="ef_construction",
        type=int,
        metavar="INT",
        default=EF_CONSTRUCTION,
        help="EF parameter for index construction",
    )
    parser.add_argument(
        "--total-bits",
        dest="total_bits",
        type=int,
        metavar="INT",
        default=TOTAL_BITS,
        help="Total number of bits for quantization",
    )
    parser.add_argument("--base-bits", dest="base_bits", type=int, metavar="INT", default=None, help="Base bits for the new x+y quantization mode (1-8). Leave unset to use --total-bits with the classic 1-bit-base scheme")
    parser.add_argument("--extra-bits", dest="extra_bits", type=int, metavar="INT", default=None, help="Extra/refinement bits for x+y quantization (0-8, default 0). Only used when --base-bits is set")
    parser.add_argument(
        "--metric",
        dest="metric",
        type=str,
        default="l2",
        choices=["l2", "ip"],
        help="Distance metric (l2 or ip)",
    )
    parser.add_argument(
        "--faster-quant",
        dest="faster_quant",
        action="store_true",
        help="Use faster quantization method",
    )
    parser.add_argument(
        "--num-threads",
        dest="num_threads",
        type=int,
        metavar="INT",
        default=NUM_THREADS,
        help="Number of threads for building the index",
    )

    args = parser.parse_args()

    main(args)
