import torch
from torch.nn import functional as F
import numpy as np


def gpu_conv2d(image, kernel):

    image_tensor = torch.tensor(image, dtype=torch.float32).cuda()
    kernel_tensor = torch.tensor(kernel, dtype=torch.float32).cuda()

    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)  # 添加输入和输出通道维度  

    out = F.conv2d(image_tensor, kernel_tensor)
    return np.real(out.cpu().numpy())