import torch
import torch.nn.functional as F


def pt_cvon2d(signal,kernel):

    # 將NumPy數轉換為PyTorch張量
    input_tensor = torch.tensor(signal, dtype=torch.double)#.cuda()
    kernel_tensor = torch.tensor(kernel, dtype=torch.double)#.cuda()

    # 執行2D卷積操作
    conv_result = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel_tensor.unsqueeze(0).unsqueeze(0))
    return conv_result
