#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os
from time import time

#WARNING: Don't try to run either of these on an actual image!!
#naive fourier transform, non optimized (for testing)
def DFTSlow(x):
    M = np.array([])
    x = np.asarray(x, dtype=float)
    n = np.arange(x.shape[0])
    N = x.shape[0]
    for k in range(x.shape[0]):
        v_k = np.exp(-2j * math.pi * k * n / N)
        M = np.concatenate((M, v_k), axis=0)
    return M.reshape(N, N).dot(x)

#naive fourier transform, optimized
def DFT(x):
    x = np.asarray(x, dtype=float)
    k = np.arange(x.shape[0]).reshape(-1, 1)
    N = x.shape[0]
    n = np.arange(N)
    M = np.exp(-2j * math.pi * k * n / N) #since k is 10x1, n is (10,): broadcasts k to 10x10 to match dims with n
    return M.dot(x)


#Fast fourier transform (my own implementation) -- slow
def FFTSlow(x):
    x = np.asarray(x, dtype=float)
    x_even = x[::2].reshape(-1, 1)
    x_odd = x[1::2].reshape(-1, 1)
    N = x.shape[0]
    m = np.arange((N/2))
    k = np.arange(N).reshape(-1, 1)

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    M_even = np.exp(-2j * math.pi * k * m / (N/2))
    result_left = M_even.dot(x_even)
    M_odd = np.exp(-2j* math.pi * k * m / (N/2))
    result_right = np.exp(-2j * math.pi * k / N) * M_odd.dot(x_odd)
    return result_left + result_right

#recursively implemented FFT -- faster
def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError(f"size of x must be a power of 2, got {N}")
    elif N <= 32:  # this cutoff should be optimized
        return DFTSlow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])




#Inverse fast fourier transform - return back from frequency space (Book)
def IFFT(x):
    x_conj = np.conj(x)
    x_new = np.fft.fft(x_conj)
    x_final = float((1.0 / len(x))) * np.conj(x_new)
    return x_final

def Runtime(method, n):
    n = int(n)
    ranges = [2**i for i in range(2, n)]
    times = []
    for i in ranges:
        x = np.random.randn(i)
        start = time()
        method(x)
        end = time() - start
        times.append(end)
    return (ranges, times)

#https://www.geeksforgeeks.org/smallest-power-of-2-greater-than-or-equal-to-n/
def nextPowerOf2(n):
    count = 0;

    # First n in the below
    # condition is for the
    # case where n is 0
    if (n and not(n & (n - 1))):
        return n

    while( n != 0):
        n >>= 1
        count += 1

    return 1 << count;


if __name__ == '__main__':
    #Naive implementation of fourier transform, check to see that it works with random input
    # x = np.random.randn(14)
    # y = DFTSlow(x)
    # print(np.allclose(y, np.fft.fft(x)))
    # print(np.allclose(DFT(x), np.fft.fft(x)))
    # print(np.allclose(FFT(x), np.fft.fft(x)))


    #actual image compression algorithm
    img = cv2.imread('cat.jpg') #make sure the image is in the same directory as this file!
    x_shape, y_shape = img.shape[0], img.shape[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f'Preprocessed Image Size: {os.path.getsize("./graycat_png.png")/1e3} KB')

    #FFT() requires vectors of length that are powers of 2, so need to zero pad the vector
    img_flat = img_gray.ravel() #flatten vector
    img_length = img_flat.shape[0]
    pad_size = nextPowerOf2(img_length) - img_length
    padding = np.zeros(pad_size)
    img_flat_padded = np.concatenate((img_flat, padding), axis=0)
    f = FFT(img_flat_padded)

    #larger C = more compression, but also more loss. C = 0 means no compression. max = 1
    C = 0.0005
    thresh = C * abs(f).max()
    ind = abs(f) > thresh
    f_thresh = f * ind
    #inverse fourier transform and convert complex to real values. Remove zero padding
    f_inv = np.abs(np.fft.ifft(f_thresh))[:img_length]
    img_back = f_inv.reshape(x_shape, y_shape)

    #Save processed image and get its file size
    cv2.imwrite('processed.png', img_back)
    print(f'Postprocessed Image Size: {os.path.getsize("./processed.png")/1e3} KB')
    #NOTE: .png format already does compression, so you won't notice a huge reduction in file size, but at thresh=0.0005*abs(f).max, it is still noticeable

    plt.subplot(121), plt.imshow(img_gray, cmap = 'gray')
    plt.title('Preprocessed Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap = 'gray')
    plt.title('Postprocessed Image'), plt.xticks([]), plt.yticks([])

    #runtime graphs of dft, fft. Uncomment to run
    # dft_x, dft_times = Runtime(DFT, np.log2(1024*32))
    # fft_x, fft_times = Runtime(FFT, np.log2(1024*32))
    # plt.subplot(2, 1, 1)
    # plt.plot(dft_x, dft_times)
    # plt.xlabel("Vector Size")
    # plt.ylabel("Execution Time (s)")
    # plt.title("DFT Runtime")
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(fft_x, fft_times)
    # plt.xlabel("Vector Size")
    # plt.ylabel("Execution Time (s)")
    # plt.title("FFT Runtime")
    # plt.tight_layout()
    plt.show()
