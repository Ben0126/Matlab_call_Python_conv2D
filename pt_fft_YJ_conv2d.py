import torch
import numpy as np
import scipy.io as spio


def pt_fft_cvon2d(signal,kernel):

    matrix1 = torch.as_tensor(signal, dtype=torch.double)#.cuda()
    matrix2 = torch.as_tensor(kernel, dtype=torch.double)#.cuda()

    # 對兩個矩陣進行快速傅立葉變換（FFT）
    fft_matrix1 = torch.fft.fft2(matrix1)
    fft_matrix2 = torch.fft.fft2(matrix2)

    # 將兩個矩陣進行逐元素相乘
    result_fft = fft_matrix1 * fft_matrix2

    # 對相乘結果進行反傅立葉變換（IFFT）
    result = torch.fft.ifft2(result_fft)

    # 將結果移回CPU並轉換為NumPy數組
    result1 = result.cpu().numpy()

    return result1