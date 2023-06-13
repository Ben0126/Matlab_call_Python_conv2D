import numpy as np
import cupy as cp
import cupyx.scipy.fft as cufft

def gpu_fft_convolve2D(signal, kernel):
    # 將數據移到GPU內存
    signal_gpu = cp.asarray(signal)
    kernel_gpu = cp.asarray(kernel)
    
    # 在GPU上進行傅立葉轉換
    signal_freq_gpu = cufft.fftn(signal_gpu)
    # 將核心陣列以0值填充，使其大小與信號陣列相同
    # padded_kernel_gpu = cp.pad(kernel_gpu, ((0, signal.shape[0] - kernel.shape[0]),
    #                     (0, signal.shape[1] - kernel.shape[1])), mode='constant')
    kernel_freq_gpu = cufft.fftn(kernel_gpu)
    
    # 在GPU上進行逐點相乘
    product_freq_gpu = signal_freq_gpu * kernel_freq_gpu
    
    # 在GPU上進行逆傅立葉轉換
    product_gpu = cufft.ifftn(product_freq_gpu)
    
    # 將結果移回內存
    product = cp.asnumpy(product_gpu)
    
    # 清除GPU的VRAM
    cp.get_default_memory_pool().free_all_blocks()

    # 返回結果
    return np.real(product)