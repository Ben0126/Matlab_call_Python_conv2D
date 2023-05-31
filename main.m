% Use matlab call python for convolution in 2D
% Here are 4 different methods,namely:
%     CPU:
%         original/FFT
%     GPU
%         original/FFT

clear;clc;
signal = randn(40, 40);
kernel = randn(40, 40);

% import py libraries
cpu_fft_conv2d = py.importlib.import_module('fft_convolve2d');
gpu_conv2d = py.importlib.import_module('pt_gpu_convolve2d');
gpu_fft_conv2d = py.importlib.import_module('gpu_fft_convolve2D');
pt_gpu_conv2d = py.importlib.import_module('pt_fft_gpu_conv2d');

%% First method, CPU original conv2D by python
% tic;
% py_result = py.scipy.signal.convolve2d(signal, kernel); 
% toc;

%% Second CPU FFT conv2D by python
tic;
result_cpu_fft = cpu_fft_conv2d.fft_conv2d(signal, kernel);
toc;
sh = double(py.array.array('d',result_cpu_fft.shape));
npary2 = double(py.array.array('d',py.numpy.nditer(result_cpu_fft)));
result_cpu_fft_mat = reshape(npary2,fliplr(sh))';  % matlab 2d array

%% Third, GPU original conv2D by python
tic;
result_gpu = gpu_conv2d.pt_conv2d(signal, kernel);
toc;

%% Fourth, GPU FFT conv2D by python
tic;
result_gpu_fft = gpu_fft_conv2d.gpu_fft_convolve2D(signal, kernel);
toc;

sh = double(py.array.array('d',result_gpu_fft.shape));
npary2 = double(py.array.array('d',py.numpy.nditer(result_gpu_fft)));
result_gpu_fft_mat = reshape(npary2,fliplr(sh))';

%% Other, GPU FFT conv2D by pytorch
% tic;
% result_pt_fft = pt_gpu_conv2d.pt_fft_von2d(signal, kernel);
% toc;

%% Disp
% disp(result_cpu_fft); 
% disp(result_cpu_fft_mat); 
% disp(result_gpu_fft); 
% disp(result_gpu_fft_mat); 
