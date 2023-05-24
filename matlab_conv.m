
input_signal = randn(1000, 1000);
kernel = randn(6, 6);

tic;
output_signal = conv2(input_signal, kernel, 'same');
toc;

tic;
py_result = py.scipy.signal.convolve2d(input_signal, kernel, 'same'); %cpu convolve2d
toc;

tic;
input_signal_fft = fft2(input_signal);
kernel_fft = fft2(kernel, size(input_signal, 1), size(input_signal, 2));
output_signal_fft = input_signal_fft .* kernel_fft;
output_signal1 = ifft2(output_signal_fft);
toc;


