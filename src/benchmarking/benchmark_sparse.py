import numpy as np
import pandas as pd
from sparse.sparse_ops import scipy_sparse_mult
import time
from typing import List, Dict
from utils.measure import measure

def run_sparse_bench(n: int, repeats: int, sparsity_levels: List[float]) -> List[Dict]:
    results = []

    for sp_level in sparsity_levels:
        density = 1.0 - sp_level
        nnz = max(1, int(n * n * density))

        A = np.zeros((n, n))
        B = np.zeros((n, n))

        idxA = np.random.choice(n * n, nnz, replace=False)
        Ai = np.unravel_index(idxA, (n, n))
        A[Ai] = np.random.rand(nnz)

        idxB = np.random.choice(n * n, nnz, replace=False)
        Bi = np.unravel_index(idxB, (n, n))
        B[Bi] = np.random.rand(nnz)

        t = measure(scipy_sparse_mult, A, B, repeats=repeats)

        results.append({
            "n": n,
            "sparsity": sp_level,
            "density": density,
            "time": t,
        })

    return results
