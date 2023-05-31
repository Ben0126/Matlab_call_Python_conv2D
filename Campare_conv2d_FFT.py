# Compare GPU and CPU at FFT convolve2D speed

import numpy as np
import time
from gpu_fft_convolve2D import gpu_fft_convolve2D
from fft_convolve2d import fft_conv2d


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

print(fft_conv_time)
print(gpu_fft_conv_time1)