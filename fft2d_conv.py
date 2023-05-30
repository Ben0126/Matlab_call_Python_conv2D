import numpy as np
import time
from multiprocessing import Pool

def conv2d(x, h):
    """計算二維卷積"""
    y = np.zeros((x.shape[0] + h.shape[0] - 1, x.shape[1] + h.shape[1] - 1))
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            xi_min = max(0, i - h.shape[0] + 1)
            xi_max = min(i + 1, x.shape[0])
            xj_min = max(0, j - h.shape[1] + 1)
            xj_max = min(j + 1, x.shape[1])
            xi = x[xi_min:xi_max, xj_min:xj_max]
            hi = h[max(0, h.shape[0]-xi.shape[0]):, max(0, h.shape[1]-xi.shape[1]):]
            y[i, j] = np.sum(xi * hi)
    return y

def convolve_serial_2d(x, h):
    """串行運算二維卷積"""
    y = conv2d(x, h)
    return y.flatten()


def convolve_parallel_2d(x, h, num_processes):
    """平行運算二維卷積"""
    y = np.zeros((x.shape[0] + h.shape[0] - 1, x.shape[1] + h.shape[1] - 1))
    pool = Pool(num_processes)
    results = []
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            xi_min = max(0, i - h.shape[0] + 1)
            xi_max = min(i + 1, x.shape[0])
            xj_min = max(0, j - h.shape[1] + 1)
            xj_max = min(j + 1, x.shape[1])
            xi = x[xi_min:xi_max, xj_min:xj_max]
            hi = h[max(0, h.shape[0]-xi.shape[0]):, max(0, h.shape[1]-xi.shape[1]):]
            results.append(pool.apply_async(conv2d, args=(xi, hi)))
    pool.close()
    pool.join()
    k = 0
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            y[i, j] = results[k].get()[0, 0]
            k += 1
    return y.flatten()


def conv2d(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
        print(imagePadded)
    else:
        imagePadded = image

    # Iterate through image
    for y in range(image.shape[1]):
        # Exit Convolution
        if y > image.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % strides == 0:
            for x in range(image.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > image.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output

def fft_convolve2d(image, kernel):
    # Pad the kernel to the size of the image
    kernel_padded = np.zeros_like(image)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    
    # Compute FFT of image and kernel
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_padded)
    
    # Compute product of FFTs and inverse FFT
    result = np.fft.ifft2(image_fft * kernel_fft)
    
    # Take real part of result (imaginary part should be very small)
    return np.real(result)

if __name__ == "__main__":
    input_signal = np.random.rand(1000, 1000)
    kernel = np.random.rand(600, 600)

    start_time = time.time()
    fft_conv = fft_convolve2d(input_signal,kernel)
    end_time = time.time()
    fft_conv_time = end_time - start_time

    start_time1 = time.time()
    conv = conv2d(input_signal,kernel)
    end_time1 = time.time()
    conv_time = end_time1 - start_time1

    start_time2 = time.time()
    y_serial = convolve_serial_2d(input_signal, kernel)
    end_time2 = time.time()
    serial_time = end_time2 - start_time2

    start_time3 = time.time()
    y_parallel = convolve_parallel_2d(input_signal, kernel, num_processes=6)
    end_time3 = time.time()
    parallel_time = end_time3 - start_time3


    print(f"FFT convolution time: {fft_conv_time:.6f} seconds")
    print(f"Normal convolution time: {conv_time:.6f} seconds")
    print(f"Serial convolution time: {serial_time:.6f} seconds")
    print(f"Parallel convolution time: {parallel_time:.6f} seconds")
