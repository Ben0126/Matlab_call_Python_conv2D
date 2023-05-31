import torch
import numpy as np
import time

def pt_conv2d(image, kernel):
   
    # 将输入数据转换为PyTorch张量
    image_tensor = torch.as_tensor(image, dtype=torch.float32).cuda()
    kernel_tensor = torch.as_tensor(kernel, dtype=torch.float32).cuda()

    # 扩展图像和卷积核的维度以适应`torch.nn.Conv2d`函数的输入要求
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)  # 添加批次和通道维度
    kernel_tensor = kernel_tensor.unsqueeze(0).unsqueeze(0)  # 添加输入和输出通道维度

    # 创建卷积层
    conv2d = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_tensor.shape[-2:], bias=False).cuda()

    # 设置卷积核权重
    conv2d.weight.data = kernel_tensor

    # 执行卷积操作
    output_tensor = conv2d(image_tensor)

    # 将结果转换回NumPy数组
    output = output_tensor.squeeze().detach().cpu().numpy()
    return np.real(output)