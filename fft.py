#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os

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


#Fast fourier transform (my own implementation) NOT WORKING xD
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

#recursively implemented FFT
def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFTSlow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])


#Inverse fast fourier transform - return back from frequency space
def IFFT(x):
    x_conj = np.conj(x)
    x_new = np.fft.fft(x_conj)
    x_final = float((1.0 / len(x))) * np.conj(x_new)
    return x_final


if __name__ == '__main__':
    #Naive implementation of fourier transform, check to see that it works with random input
    x = np.random.randn(16)
    y = DFTSlow(x)
    print(np.allclose(y, np.fft.fft(x)))
    print(np.allclose(DFT(x), np.fft.fft(x)))
    print(np.allclose(FFT(x), np.fft.fft(x)))


    #actual image compression stuff
    img = cv2.imread('cat.jpg')
    x_shape, y_shape = img.shape[0], img.shape[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    print(f'Preprocessed Image Size: {os.path.getsize("./graycat_png.png")/1e3} KB')

    #Compression algorithm. Remove frequencies lower than the threshold
    #TODO: replace np.fft.fft() with our own fft implementation
    f = np.fft.fft(img_gray.reshape(-1, 1))
    #larger thresh = more compression, but also more loss. thresh = 0 means no compression. max = 1
    thresh = 0.5 * abs(f).max()
    ind = abs(f) > thresh
    f_thresh = f * ind
    #inverse fourier transform and convert complex to real values
    f_inv = np.abs(np.fft.ifft(f_thresh))
    img_back = f_inv.reshape(x_shape, y_shape)

    #Save processed image and get its file size
    cv2.imwrite('processed.png', img_back)
    #TODO: get algorithm to work with colored images
    print(f'Postprocessed Image Size: {os.path.getsize("./processed.png")/1e3} KB')
    #NOTE: .png format already does compression, so you won't notice a huge reduction in file size, but at thresh=0.5*abs(f).max, it is still noticeable

    plt.subplot(121), plt.imshow(img_gray, cmap = 'gray')
    plt.title('Preprocessed Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img_back, cmap = 'gray')
    plt.title('Postprocessed Image'), plt.xticks([]), plt.yticks([])
    plt.show()
