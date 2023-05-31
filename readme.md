# Comparing GPU and CPU Speed for FFT Convolution

This repository provides code for comparing the speed of GPU and CPU implementations for FFT convolution. The code compares the execution time for performing 2D convolution using FFT on both GPU and CPU.

## Usage

To run the code, follow these steps:

1. Make sure you have Python installed on your system.
2. Clone this repository or download the Python script.
3. Open a terminal or command prompt and navigate to the directory where the script is located.
4. Run the script using the Python interpreter.

The script includes the following steps:

1. Generate random input signal and kernel matrices of size 5000x5000.
2. Perform FFT convolution on the CPU using the `fft_conv2d` function and measure the execution time.
3. Perform FFT convolution on the GPU using the `gpu_fft_convolve2D` function and measure the execution time.
4. Perform FFT convolution on the GPU using the `pt_fft_von2d` function (using PyTorch) and measure the execution time.
5. Print the execution times for each method.

The results will be displayed in the terminal or command prompt, showing the execution times for CPU conv2d, GPU FFT, and GPU pt FFT methods.

Feel free to modify the code and experiment with different input signal and kernel sizes to observe the speed differences between GPU and CPU implementations.

## Requirements

Make sure you have the following Python libraries installed before running the code:

- `numpy`
- `time`
- `gpu_fft_convolve2D`
- `fft_conv2d`
- `pt_fft_von2d`

## License

This project is licensed under the [MIT License](LICENSE).

