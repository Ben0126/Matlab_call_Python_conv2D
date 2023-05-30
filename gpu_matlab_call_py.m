clear; clc;
signal = randn(1000, 1000);
kernel = randn(600, 600);

%pyrun("import numpy as np")
pyrunfile(fft2d_conv.py)

%% Matlab 內建conv2D
tic;
output_signal = conv2(signal, kernel, 'same');
toc;

%% Matlab FFT conv2D
tic;
input_signal_fft = fft2(signal);
kernel_fft = fft2(kernel, size(signal, 1), size(signal, 2));
output_signal_fft = input_signal_fft .* kernel_fft;
output_signal1 = ifft2(output_signal_fft);
toc;

%% Python conv2D
%tic;
%py_result = py.scipy.signal.convolve2d(signal, kernel, 'same'); 
%toc;

%% Python GPU FFT conv2D
tic;
py.importlib.import_module('gpu_fft_convolve2D')
result = py.gpu_fft_convolve2D.gpu_fft_convolve2D(signal, kernel);
toc;

%disp(output_signal); %matlab conv2d
%disp(output_signal1); %matlab fft
%disp(py_result); %py conv2d
%disp(result); %py GPU FFT






