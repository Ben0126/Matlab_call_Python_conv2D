# Compare GPU and CPU at FFT convolve2D speed

import numpy as np
import time
from gpu_fft_convolve2D import gpu_fft_convolve2D
from fft_convolve2d import fft_conv2d
from pt_fft_gpu_conv2d import pt_fft_von2d


a = 4000
b = 4000
input_signal = np.random.rand(a, a)
kernel = np.random.rand(b, b)

start_time = time.time()
fft_conv = fft_conv2d(input_signal,kernel)
end_time = time.time()
fft_conv_time = end_time - start_time

start_time1 = time.time()
fft_conv = gpu_fft_convolve2D(input_signal,kernel)
end_time1 = time.time()
gpu_fft_conv_time1 = end_time1 - start_time1

start_time2 = time.time()
pt_fft_conv = pt_fft_von2d(input_signal,kernel)
end_time2 = time.time()
gpu_fft_conv_time2 = end_time2 - start_time2


print(f"CPU conv2d time: {fft_conv_time:.6f} seconds")
print(f"GPU FFT time: {gpu_fft_conv_time1:.6f} seconds")
print(f"GPU pyTorch FFT time: {gpu_fft_conv_time2:.6f} seconds")