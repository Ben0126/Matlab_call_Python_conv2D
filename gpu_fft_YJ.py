import cupy as cp
import cupyx.scipy.fft as cufft
import numpy as np
import time


size = 10000
signal = np.random.rand(size, size)
kernel = np.random.rand(size, size)
start_time = time.time()
# 將數據移動到GPU內存
signal_gpu = cp.asarray(signal)
kernel_gpu = cp.asarray(kernel)

# 將核心陣列以零值填充，使其大小與信號陣列相同
padded_kernel_gpu = cp.pad(kernel_gpu, ((0, signal.shape[0] - kernel.shape[0]), 
                        (0, signal.shape[1] - kernel.shape[1])), mode='constant')
kernel_freq_gpu = cufft.fftn(padded_kernel_gpu)


# 在GPU上進行傅立葉轉換
signal_freq_gpu = cufft.fftn(signal_gpu)

# 在GPU上進行逐點相乘
product_freq_gpu = signal_freq_gpu * kernel_freq_gpu

# 在GPU上進行逆傅立葉轉換
product_gpu = cufft.ifftn(product_freq_gpu)
end_time = time.time()
fft_conv_time = end_time - start_time
# 將結果移回主機內存
product = cp.asnumpy(product_gpu)

# 返回結果
print(f"GPU FFT time: {fft_conv_time:.6f} seconds")