# Convolution in 2D using Python and MATLAB
For more detailed information about comparing GPU and CPU speed for FFT convolution, please refer to the [Compare GPU and CPU Speed for FFT Convolution README](link-to-second-readme).

This repository provides MATLAB code for performing 2D convolution using Python. It demonstrates four different methods for convolution, including both CPU and GPU implementations. The methods are as follows:

1. **CPU - Original Convolution**: This method performs 2D convolution using the original conv2D function in Python.

2. **CPU - FFT Convolution**: This method uses the Fast Fourier Transform (FFT) algorithm to perform 2D convolution on the CPU.

3. **GPU - Original Convolution**: This method utilizes the GPU to perform 2D convolution using Python.

4. **GPU - FFT Convolution**: This method employs the GPU and FFT algorithm to perform 2D convolution.

To run the code, follow these steps:

1. Make sure you have MATLAB installed on your system.

2. Clone this repository or download the MATLAB script.

3. Open MATLAB and navigate to the directory where the script is located.

4. Run the script in MATLAB.

The script includes commented sections for each method. You can uncomment the desired method to perform the corresponding convolution. By default, the second and fourth methods (CPU FFT and GPU FFT convolution) are enabled.

Note: The code requires specific Python libraries, which are imported using the `py` module in MATLAB. Make sure you have these libraries installed in your Python environment.

Feel free to modify the code and experiment with different inputs to observe the performance differences between CPU and GPU convolution methods.

Enjoy!

## License

This project is licensed under the [MIT License](LICENSE)
