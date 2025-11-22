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
        results_measure = measure(scipy_sparse_mult, A, B, repeats=repeats)

        if results_measure:
            avg_time = sum(results_measure["times"]) / len(results_measure["times"]) if results_measure["times"] else None
            avg_cpu = sum(results_measure["cpu"]) / len(results_measure["cpu"]) if results_measure["cpu"] else None
            avg_mem = sum(results_measure["memory"]) / len(results_measure["memory"]) if results_measure["memory"] else None
        else:
            avg_time = avg_cpu = avg_mem = None

        results.append({
            "n": n,
            "sparsity": sp_level,
            "density": density,
            "time": avg_time,
            "cpu": avg_cpu,
            "memory": avg_mem,
        })

    return results
