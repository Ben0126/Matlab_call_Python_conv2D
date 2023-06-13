import torch
import numpy as np
import time
import scipy.io as spio

size = 4000
signal = np.random.rand(size, size)
kernel = np.random.rand(size, size)

start_time = time.time()
# signal = spio.loadmat('kernel.mat')['kernel']
# kernel = spio.loadmat('signal.mat')['signal']
matrix1 = torch.as_tensor(signal, dtype=torch.float64).cuda()
matrix2 = torch.as_tensor(kernel, dtype=torch.float64).cuda()

# 對兩個矩陣進行快速傅立葉變換（FFT）
fft_matrix1 = torch.fft.fft2(matrix1)
fft_matrix2 = torch.fft.fft2(matrix2)

# 將兩個矩陣進行逐元素相乘
result_fft = fft_matrix1 * fft_matrix2

# 對相乘結果進行反傅立葉變換（IFFT）
result = torch.fft.ifft2(result_fft)

# 將結果移回CPU並轉換為NumPy數組
result1 = result.cpu().numpy()
end_time = time.time()

fft_conv_time = end_time - start_time

print(result1)
print(f"GPU FFT TRY time: {fft_conv_time:.6f} seconds")
