# 2D Convolution using MATLAB and Python

...

For more detailed information about comparing GPU and CPU speed for FFT convolution, please refer to the [Compare GPU and CPU Speed for FFT Convolution README](link-to-second-readme).

...



This repository provides MATLAB code for performing 2D convolution using Python. It demonstrates four different methods for convolution, including both CPU and GPU implementations. The methods are as follows:

## CPU:
1. **Original Convolution**: This method performs 2D convolution using the original conv2D function in Python.
2. **FFT Convolution**: This method uses the Fast Fourier Transform (FFT) algorithm to perform 2D convolution on the CPU.

## GPU:
1. **Original Convolution**: This method utilizes the GPU to perform 2D convolution using Python.
2. **FFT Convolution**: This method employs the GPU and FFT algorithm to perform 2D convolution.

To run the code, follow these steps:

1. Make sure you have MATLAB installed on your system.
2. Clone this repository or download the MATLAB script.
3. Open MATLAB and navigate to the directory where the script is located.
4. Run the script in MATLAB.

The script includes commented sections for each method. You can uncomment the desired method to perform the corresponding convolution. By default, the second and fourth methods (CPU FFT and GPU FFT convolution) are enabled.

**Note:** Currently, there is an issue with the third method (GPU original convolution) when using PyTorch in MATLAB, resulting in the error message "Python Error: ValueError: could not determine the shape of object type 'memoryview'." This issue is still pending resolution. Therefore, the third method is not executable in MATLAB at the moment.

The resulting convolutions are stored in MATLAB variables `result_cpu_fft_mat` and `result_gpu_fft_mat` for the CPU FFT and GPU FFT methods, respectively.

Feel free to modify the code and experiment with different input signals and kernels to observe the convolution results using different methods.

Enjoy!

**Note:** The `disp` statements at the end of the script can be uncommented to display the convolution results for each method.
