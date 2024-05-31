import sys
import cv2
import numpy as np

def cross_correlation_2d(img, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    m, n = kernel.shape

    if (len(img.shape) == 3):
        h, w, ch = img.shape
    else:
        h, w = img.shape
        ch = 1
        img = np.expand_dims(img, axis=2)

    out = np.zeros_like(img, dtype=img.dtype)  
   
    # pad the image above and below by m/2 with zeros
    # pad the image to the left and right by n/2 zeros
    a = m//2
    b = m//2 + h
    c = n//2
    d = n//2 + w
    padded_img = np.zeros((h + m - 1, w + n - 1, ch), dtype=img.dtype)
    padded_img[a:b,c:d,:] = img
    
    # do calcs
    for i in range(h):
        for j in range(w):
            # select the subportion from padded_image
            a = m//2 + i - m//2
            b = m//2 + i + m//2 + 1
            c = n//2 + j - n//2
            d = n//2 + j + n//2 + 1
            working_padded_img = padded_img[a:b,c:d,:]

            for k in range(ch):
                submat = working_padded_img[:,:,k]
                kernel_array = np.reshape(kernel, -1)
                img_array = np.reshape(submat,-1)
                out[a,c,k] = np.dot(kernel_array,img_array)

    if (ch == 1):
        fin_out = out.reshape(h,w)
        return fin_out
    else:
        return out

    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    kernel_flip_ud = np.flipud(kernel)
    kernel_fin = np.fliplr(kernel_flip_ud)
    return cross_correlation_2d(img,kernel_fin)
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")

    x_mid, y_mid = width // 2, height // 2
    ind_x = np.arange(-x_mid, x_mid + 1)
    ind_y = np.arange(-y_mid, y_mid + 1)

    X = np.empty(width)
    for i, x in enumerate(ind_x):
        X[i] = x**2

    Y = np.empty(height)
    for i, y in enumerate(ind_y):
        Y[i] = y**2

    X = np.exp(-X / (2 * sigma **2))
    Y = np.exp(-Y / (2 * sigma **2)) / (2 * sigma**2 * np.pi)
    
    output = np.empty((height, width))
    for i, y_val in enumerate(Y):
        for j, x_val in enumerate(X):
            output[i, j] = x_val * y_val

    normalize = np.sum(Y) * np.sum(X)
    output /= normalize
    
    return output
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    # raise Exception("TODO in hybrid.py not implemented")
    return img - low_pass(img, sigma, size)
    # TODO-BLOCK-END

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
        high_low2, mixin_ratio, scale_factor):
    '''This function adds two images to create a hybrid image, based on
    parameters specified by the user.'''
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *=  (1 - mixin_ratio)
    img2 *= mixin_ratio
    hybrid_img = (img1 + img2) * scale_factor
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

