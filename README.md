# 2D Convolution using MATLAB and Python

...
## [Compare_GPU_CPU_FFT_Convolution](FFT_Convolutio-readme.md)
If you are interested in comparing the performance of FFT convolution on GPU and CPU using Python.

...

This repository provides MATLAB code for performing 2D convolution using Python. It demonstrates four different methods for convolution, including both CPU and GPU implementations. The methods are as follows:

## CPU:
1. **Original Convolution:** This method performs 2D convolution using the original conv2D function in Python.
2. **FFT Convolution:** This method uses the Fast Fourier Transform (FFT) algorithm to perform 2D convolution on the CPU.

## GPU:
1. **Original Convolution:** This method utilizes the GPU to perform 2D convolution using Python.
2. **FFT Convolution:** This method employs the GPU and FFT algorithm to perform 2D convolution.

The script includes commented sections for each method. You can uncomment the desired method to perform the corresponding convolution. By default, the second and fourth methods (CPU FFT and GPU FFT convolution) are enabled.

**Note:** Currently, there is an issue with the third method (GPU original convolution) when using PyTorch in MATLAB, resulting in the error message "Python Error: ValueError: could not determine the shape of object type 'memoryview'." This issue is still pending resolution. Therefore, the third method is not executable in MATLAB at the moment.

The resulting convolutions are stored in MATLAB variables **'result_cpu_fft_mat'** and **'result_gpu_fft_mat'** for the CPU FFT and GPU FFT methods, respectively.

Feel free to modify the code and experiment with different input signals and kernels to observe the convolution results using different methods.

## Enjoy!

Note: The 'disp' statements at the end of the script can be uncommented to display the convolution results for each method.

## [matlab_mex_test link](https://github.com/andy856996/parallel-computing-cuda/tree/main/matlab_mex_test)
If you want to learn more about the performance of convolution on CUDA, C, and MATLAB.
