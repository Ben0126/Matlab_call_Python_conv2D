import numpy as np
import time
from multiprocessing import Pool

def conv(x, h):
    """計算卷積"""
    y = np.convolve(x, h)

def convolve_serial(x, h):
    """串行運算卷積"""
    x_len = len(x)
    h_len = len(h)
    y_len = x_len + h_len - 1
    y = np.zeros(y_len)
    for i in range(y_len):
        start = max(0, i - h_len + 1)
        end = min(i + 1, x_len)
        xi = x[start:end]
        hi = h[max(0, len(xi)-h_len):]
        y[i] = conv(xi, hi)
    return y

def convolve_parallel(x, h, num_processes):
    """平行運算卷積"""
    x_len = len(x)
    h_len = len(h)
    y_len = x_len + h_len - 1
    y = np.zeros(y_len)
    pool = Pool(num_processes)
    results = []
    for i in range(y_len):
        start = max(0, i - h_len + 1)
        end = min(i + 1, x_len)
        xi = x[start:end]
        hi = h[0:i-start+1]
        results.append(pool.apply_async(conv, args=(xi, hi)))
    pool.close()
    pool.join()
    for i, res in enumerate(results):
        y[i] = res.get()
    return y

# 測試

if __name__=="__main__":
    x = np.random.rand(10000)
    h = np.random.rand(10000)
    
    start_time = time.time()
    y_serial = convolve_serial(x, h)
    end_time = time.time()
    serial_time = end_time - start_time

    start_time = time.time()
    y_parallel = convolve_parallel(x, h, num_processes=6)
    end_time = time.time()
    parallel_time = end_time - start_time

    print(f"Serial convolution time: {serial_time:.6f} seconds")
    print(f"Parallel convolution time: {parallel_time:.6f} seconds")
    print(f"Speedup: {serial_time/parallel_time:.2f}x")
