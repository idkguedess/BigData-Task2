import numpy as np
import time
import pandas as pd
from algorithms.naive import multiply_matrices
from algorithms.blocked import blocked_dot
from algorithms.numpy_dot import numpy_dot_mult
from typing import List, Dict
from utils.measure import measure
from utils.generate_matrices import allocate_matrix, initialize_matrix

def run_dense_bench(size: int, repeats: int, algorithms: List[str]) -> List[Dict]:
    results = []

    A = np.random.rand(size, size)
    B = np.random.rand(size, size)

    for alg in algorithms:
        row = {"n": size, "algorithm": alg, "time": None, "cpu": None, "memory": None}

        if alg == "numpy":
            meas = measure(numpy_dot_mult, A, B, repeats=repeats)
            if meas:
                row["time"] = sum(meas["times"]) / len(meas["times"]) if meas["times"] else None
                row["cpu"] = sum(meas["cpu"]) / len(meas["cpu"]) if meas["cpu"] else None
                row["memory"] = sum(meas["memory"]) / len(meas["memory"]) if meas["memory"] else None
            else:
                row["time"] = row["cpu"] = row["memory"] = None

        elif alg == "blocked":
            meas = measure(blocked_dot, A, B, repeats=repeats)
            if meas:
                row["time"] = sum(meas["times"]) / len(meas["times"]) if meas["times"] else None
                row["cpu"] = sum(meas["cpu"]) / len(meas["cpu"]) if meas["cpu"] else None
                row["memory"] = sum(meas["memory"]) / len(meas["memory"]) if meas["memory"] else None
            else:
                row["time"] = row["cpu"] = row["memory"] = None

        elif alg == "naive":
            a = allocate_matrix(size)
            b = allocate_matrix(size)
            c = allocate_matrix(size)
            initialize_matrix(a, size, seed=42)
            initialize_matrix(b, size, seed=123)

            def run_naive():
                multiply_matrices(a, b, c, size)

            results_measure = measure(lambda: run_naive(), repeats=repeats)
            if results_measure:
                row["time"] = sum(results_measure["times"]) / len(results_measure["times"]) if results_measure["times"] else None
                row["cpu"] = sum(results_measure["cpu"]) / len(results_measure["cpu"]) if results_measure["cpu"] else None
                row["memory"] = sum(results_measure["memory"]) / len(results_measure["memory"]) if results_measure["memory"] else None
            else:
                row["time"] = row["cpu"] = row["memory"] = None

        else:
            raise ValueError(f"Unknown algorithm: {alg}")

        results.append(row)

    return results