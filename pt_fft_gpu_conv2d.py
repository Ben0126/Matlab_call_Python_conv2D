import numpy as np
import torch
import torch.nn.functional as F


def pt_fft_cvon2d(signal,kernel):

    # 将NumPy数组转换为PyTorch张量
    input_tensor = torch.tensor(signal, dtype=torch.double)
    kernel_tensor = torch.tensor(kernel, dtype=torch.double)

    # 执行2D卷积操作
    conv_result = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel_tensor.unsqueeze(0).unsqueeze(0))
    return conv_result
