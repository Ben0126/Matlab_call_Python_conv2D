import numpy as np
import cupy as cp
import time

def convolve2d_gpu(image, kernel):
    # 將 NumPy 陣列轉換為 CuPy 陣列
    image_gpu = cp.array(image)
    kernel_gpu = cp.array(kernel)

    # 獲取圖像和內核的形狀
    image_shape = image_gpu.shape
    kernel_shape = kernel_gpu.shape

    # 計算輸出圖像的形狀
    output_shape = (image_shape[0] - kernel_shape[0] + 1, image_shape[1] - kernel_shape[1] + 1)

    # 初始化輸出圖像
    output_gpu = cp.zeros(output_shape)

    # 進行卷積操作
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output_gpu[i, j] = cp.sum(image_gpu[i:i + kernel_shape[0], j:j + kernel_shape[1]] * kernel_gpu)

    # 將結果從 GPU 轉回 CPU
    output = cp.asnumpy(output_gpu)

    return output

# 測試
image = np.random.rand(10000, 10000)
kernel = np.random.rand(6000, 6000)


t1 = time.time()
output = convolve2d_gpu(image, kernel)
t2 = time.time()
tt = t2-t1
print(tt)