% Use matlab call python for convolution in 2D
% Here are 4 different methods,namely:
%     CPU:
%         original/FFT
%     GPU
%         original/FFT

%%
clear; clc;
signal = randn(1000, 1000);
kernel = randn(600, 600);

%% import py libraries
gpu_fft_conv2d = py.importlib.import_module('gpu_fft_convolve2D');

%% Python CPU original conv2D 
% tic;
% py_result = py.scipy.signal.convolve2d(signal, kernel); 
% toc;

%% Python CPU FFT conv2D 


%% Python GPU original conv2D


%% Python GPU FFT conv2D
tic;
result = gpu_fft_conv2d.gpu_fft_convolve2D(signal, kernel);
toc;

