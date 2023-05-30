% Use matlab call python for convolution in 2D
% Here are 4 different methods,namely:
%     CPU:
%         original/FFT
%     GPU
%         original/FFT

clear;clc;
signal = randn(10000, 10000);
kernel = randn(6000, 6000);

% signal = ones(10000, 10000);
% kernel = ones(6000, 6000);

% import py libraries
cpu_fft_conv2d = py.importlib.import_module('fft_convolve2d');
gpu_fft_conv2d = py.importlib.import_module('gpu_fft_convolve2D');


%% First method, CPU original conv2D by python
% tic;
% py_result = py.scipy.signal.convolve2d(signal, kernel); 
% toc;

%% Second CPU FFT conv2D by python
tic;
result_cpu_fft = cpu_fft_conv2d.fft_conv2d(signal, kernel);
toc;

%% Third, GPU original conv2D by python


%% Fourth, GPU FFT conv2D by python
tic;
result_gpu_fft = gpu_fft_conv2d.gpu_fft_convolve2D(signal, kernel);
toc;

