import numpy as np
import time
import pandas as pd
from algorithms.naive import naive_python_mult
from algorithms.blocked import blocked_dot
from algorithms.numpy_dot import numpy_dot_mult

def measure(func, *args, repeats=1, timeout=20):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(*args)
        t1 = time.perf_counter()
        if t1 - t0 > timeout:
            return None
        times.append(t1 - t0)
    return sum(times) / repeats

def benchmark_dense(sizes=[64,128,256,512]):
    results = []

    for n in sizes:
        A = np.random.rand(n,n)
        B = np.random.rand(n,n)

        row = {"n": n}

        row["numpy_dot"] = measure(numpy_dot_mult, A, B)
        row["blocked"]   = measure(blocked_dot, A, B, block=64)

        if n <= 128:
            row["naive"] = measure(naive_python_mult, A, B)
        else:
            row["naive"] = None

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv("dense_results.csv", index=False)
    print(df)

if __name__ == "__main__":
    benchmark_dense()
