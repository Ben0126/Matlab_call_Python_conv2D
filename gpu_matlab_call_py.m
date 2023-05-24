clear; clc;
signal = randn(1000, 1000);
kernel = randn(6, 6);

tic;
py.importlib.import_module('gpu_fft_convolve2D')
pyrun("import numpy as np")
result = py.gpu_fft_convolve2D.gpu_fft_convolve2D(signal, kernel);
toc;

% 将结果转换为MATLAB数组
%result = double(result);

% 打印结果
disp(result);
