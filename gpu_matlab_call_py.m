clear; clc;
signal = randn(100, 100);
kernel = randn(60, 60);

% pyrun("import numpy as np")
% pyrun("import cupy as cp")
% pyrun("import cupyx.scipy.fft as cufft")

%% Matlab 內建conv2D
tic;
output_signal = conv2(signal, kernel);
toc;

%% Matlab FFT conv2D
tic;
input_signal_fft = fft2(signal);
kernel_fft = fft2(kernel, size(signal, 1), size(signal, 2));
output_signal_fft = input_signal_fft .* kernel_fft;
output_signal1 = ifft2(output_signal_fft);
toc;

%% Third, GPU original conv2D by python
tic;
gpu_conv2d = py.importlib.import_module('pt_gpu_convolve2d');
result_gpu = gpu_conv2d.pt_conv2d(signal, kernel);
toc;

%% Python GPU FFT conv2D
tic;
py.importlib.import_module('gpu_fft_convolve2D')
py.gpu_fft_convolve2D.gpu_fft_convolve2D(signal, kernel);
toc;

%disp(output_signal); %matlab conv2d
%disp(output_signal1); %matlab fft
%disp(py_result); %py conv2d
%disp(result); %py GPU FFT






