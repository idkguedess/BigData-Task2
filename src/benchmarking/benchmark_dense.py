import numpy as np
import time
import pandas as pd
from algorithms.naive import naive_python_mult
from algorithms.blocked import blocked_dot
from algorithms.numpy_dot import numpy_dot_mult
from algorithms import multiply_matrices, allocate_matrix, initialize_matrix
from typing import List, Dict
from utils.measure import measure

def run_dense_bench(size: int, repeats: int, algorithms: List[str]) -> List[Dict]:
    results = []

    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    for alg in algorithms:
        row = {"n": size, "algorithm": alg, "time": None}

        if alg == "numpy":
            t = measure(numpy_dot_mult, A, B, repeats=repeats)
            row["time"] = t

        elif alg == "blocked":
            t = measure(blocked_dot, A, B, repeats=repeats)
            row["time"] = t

        elif alg == "naive":
            a = allocate_matrix(size)
            b = allocate_matrix(size)
            c = allocate_matrix(size)
            initialize_matrix(a, size, seed=42)
            initialize_matrix(b, size, seed=123)

            def run_naive():
                multiply_matrices(a, b, c, size)

            t = measure(lambda: run_naive(), repeats=repeats)
            row["time"] = t

        else:
            raise ValueError(f"Unknown algorithm: {alg}")

        results.append(row)

    return results