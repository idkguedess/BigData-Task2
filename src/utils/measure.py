from time import time


def measure(fn, *args, repeats: int = 1, timeout: float = 60.0):
    """Run `fn(*args)` `repeats` times and return average time in seconds.
    If any run exceeds `timeout`, return None for that measurement.
    """
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        dt = t1 - t0
        if dt > timeout:
            return None
        times.append(dt)
    return sum(times) / len(times) if times else None