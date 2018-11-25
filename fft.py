#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os

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



def FFT(x):
    pass

def IFFT(x):
    pass


if __name__ == '__main__':
    #Naive implementation of fourier transform, check to see that it works with random input
    x = np.random.randn(10)
    y = DFTSlow(x)
    print(np.allclose(y, np.fft.fft(x)))
    print(np.allclose(DFT(x), np.fft.fft(x)))

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
