import time
import psutil

def measure(fn, *args, repeats: int = 1, timeout: float = 60.0):
    """Run `fn(*args)` `repeats` times and return average time in seconds.
    If any run exceeds `timeout`, return None for that measurement.
    """
    results = {"times": [], "cpu": [], "memory": []}
    process = psutil.Process()
    for repeat in range(repeats):
        cpu_start = psutil.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024)
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        cpu_end = psutil.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024)
        elapsed = t1 - t0
        avg_cpu = (cpu_start + cpu_end) / 2
        mem_used = mem_after - mem_before
        results["times"].append(elapsed)
        results["cpu"].append(avg_cpu)
        results["memory"].append(mem_used)
        print(f"{repeat + 1},{elapsed:.6f},{avg_cpu:.2f},{mem_used:.2f}")
    return results