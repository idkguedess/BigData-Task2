import numpy as np
import pandas as pd
from sparse.sparse_ops import scipy_sparse_mult
import time

def measure(func, *args, timeout=30):
    t0 = time.perf_counter()
    func(*args)
    t1 = time.perf_counter()
    if t1 - t0 > timeout:
        return None
    return t1 - t0

def benchmark_sparse(n=1024, sparsity_levels=[0.9,0.95,0.99,0.995]):
    results = []

    for sp_level in sparsity_levels:
        density = 1 - sp_level
        nnz = max(1, int(n*n*density))

        A = np.zeros((n,n))
        B = np.zeros((n,n))

        idxA = np.random.choice(n*n, nnz, replace=False)
        Ai = np.unravel_index(idxA, (n,n))
        A[Ai] = np.random.rand(nnz)

        idxB = np.random.choice(n*n, nnz, replace=False)
        Bi = np.unravel_index(idxB, (n,n))
        B[Bi] = np.random.rand(nnz)

        t = measure(scipy_sparse_mult, A, B)

        results.append({
            "n": n,
            "sparsity": sp_level,
            "density": density,
            "time": t
        })

    df = pd.DataFrame(results)
    df.to_csv("sparse_results.csv", index=False)
    print(df)

if __name__ == "__main__":
    benchmark_sparse()
