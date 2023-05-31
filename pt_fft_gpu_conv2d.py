import numpy as np
import torch
import torch.nn.functional as F
import time

def pt_fft_von2d(signal,kernel):

    # 将NumPy数组转换为PyTorch张量
    input_tensor = torch.tensor(signal, dtype=torch.float32)
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32)

    # 执行2D傅立叶变换
    fft_result = torch.fft.fft2(input_tensor, dim=(-2, -1))

    # 执行2D卷积操作
    conv_result = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel_tensor.unsqueeze(0).unsqueeze(0))
    return conv_result


# 生成输入张量和卷积核张量
input_signal = np.random.rand(4000, 4000)
kernel = np.random.rand(4000, 4000)

start_time1 = time.time()
r = pt_fft_von2d(input_signal,kernel)
end_time1 = time.time()

gpu_fft_conv_time1 = end_time1 - start_time1
print(gpu_fft_conv_time1)




