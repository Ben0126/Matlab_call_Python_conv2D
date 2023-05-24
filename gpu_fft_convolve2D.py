import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft
import time

def gpu_fft_convolve2D(signal, kernel):
    # 將數據移動到GPU內存
    signal_gpu = cp.asarray(signal)
    kernel_gpu = cp.asarray(kernel)
    
    # 在GPU上進行傅立葉轉換
    signal_freq_gpu = cufft.fftn(signal_gpu)
    
    # Pad the kernel array with zeros to match the size of the signal array
    padded_kernel_gpu = cp.pad(kernel_gpu, ((0, signal.shape[0] - kernel.shape[0]), (0, signal.shape[1] - kernel.shape[1])), mode='constant')
    
    kernel_freq_gpu = cufft.fftn(padded_kernel_gpu)
    
    # 在GPU上進行逐點相乘
    product_freq_gpu = signal_freq_gpu * kernel_freq_gpu
    
    # 在GPU上進行逆傅立葉轉換
    product_gpu = cufft.ifftn(product_freq_gpu)
    
    # 將結果移回主機內存
    product = cp.asnumpy(product_gpu)
    
    # 返回結果
    return np.real(product)

# # 創建2D信號和卷積核
# signal = np.random.rand(1000, 1000)  # 10000x10000的隨機信號
# kernel = np.random.rand(600, 600)  

# t1 = time.time()
# # 使用GPU加速的傅立葉卷積計算
# result = gpu_fft_convolve2D(signal, kernel)
# t2 = time.time()

# t = t2-t1
# # 打印結果
# print(result)
