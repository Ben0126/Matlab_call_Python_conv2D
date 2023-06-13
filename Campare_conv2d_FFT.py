import numpy as np
import time
import matplotlib.pyplot as plt
from gpu_fft_convolve2D import gpu_fft_convolve2D
from fft_convolve2d import fft_conv2d
from pt_gpu_conv2d import pt_cvon2d
from pt_fft_YJ_conv2d import pt_fft_cvon2d


sizes = [100,500,1000,4000,7000,10000]
cpu_times = []
gpu_fft_times = []
pt_times = []
pt_fft_times = []

for size in sizes:
    input_signal = np.random.rand(size, size)
    kernel = np.random.rand(size, size)

    # start_time = time.time()
    # fft_conv = fft_conv2d(input_signal, kernel)
    # end_time = time.time()
    # fft_conv_time = end_time - start_time
    # cpu_times.append(fft_conv_time)

    start_time1 = time.time()
    gpu_fft_conv = gpu_fft_convolve2D(input_signal, kernel)
    end_time1 = time.time()
    gpu_fft_conv_time = end_time1 - start_time1
    gpu_fft_times.append(gpu_fft_conv_time)

    # start_time2 = time.time()
    # pt_conv = pt_cvon2d(input_signal, kernel)
    # end_time2 = time.time()
    # pt_time = end_time2 - start_time2
    # pt_times.append(pt_time)

    start_time3 = time.time()
    pt_fft_conv = pt_fft_cvon2d(input_signal, kernel)
    end_time3 = time.time()
    pt_fft_conv_time = end_time3 - start_time3
    pt_fft_times.append(pt_fft_conv_time)

    print(f"Size: {size} x {size}")
    # print(f"CPU conv2d time: {fft_conv_time:.6f} seconds")
    print(f"GPU FFT time: {gpu_fft_conv_time:.6f} seconds")
    # print(f"GPU pyTorch time: {pt_time:.6f} seconds")
    print(f"GPU FFT pyTorch time: {pt_fft_conv_time:.6f} seconds")