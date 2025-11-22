import argparse
import os
from pathlib import Path

from benchmarking.benchmark_dense import run_dense_bench
from benchmarking.benchmark_sparse import run_sparse_bench
from utils.csv_writer import write_csv

def parse_args():
    DEFAULT_OUT = str(Path(__file__).resolve().parents[1] / "benchmarks")

    parser = argparse.ArgumentParser(description="Run dense and sparse matrix benchmarks")
    parser.add_argument("--size", "-s", type=int, default=256, help="Matrix size (n for n x n)")
    parser.add_argument("--repeats", "-n", type=int, default=3, help="Number of repeats per measurement")
    parser.add_argument("--algorithms", "-a", nargs="*", default=["numpy", "blocked", "naive"],
                        choices=["numpy", "blocked", "naive"], help="Algorithms to benchmark")
    parser.add_argument("--sparse", action="store_true", default=True, help="Also run sparse benchmarks")
    parser.add_argument("--sparse-size", type=int, default=1024, help="Size for sparse benchmarks (n used in sparse) ")
    parser.add_argument("--sparsity-levels", nargs="*", type=float, default=[0.9, 0.95, 0.99, 0.995],
                        help="List of sparsity levels to test (values between 0 and 1, e.g. 0.95)")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT, help="Directory to write results CSVs")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Running dense benchmarks: size={args.size}, repeats={args.repeats}, algorithms={args.algorithms}")
    dense_results = run_dense_bench(args.size, args.repeats, args.algorithms)
    dense_csv = os.path.join(args.out_dir, "dense_results_from_main.csv")
    write_csv(dense_csv, dense_results)
    print(f"Saved dense results to {dense_csv}")

    if args.sparse:
        print(f"Running sparse benchmarks: n={args.sparse_size}, repeats={args.repeats}, sparsity_levels={args.sparsity_levels}")
        sparse_results = run_sparse_bench(args.sparse_size, args.repeats, args.sparsity_levels)
        sparse_csv = os.path.join(args.out_dir, "sparse_results_from_main.csv")
        write_csv(sparse_csv, sparse_results)
        print(f"Saved sparse results to {sparse_csv}")


if __name__ == "__main__":
    main()
