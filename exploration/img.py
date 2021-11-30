import numpy as np

import cv2

from scipy import signal
from scipy import misc
import skimage
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line
from skimage import data

from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
from matplotlib import cm

scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

def remove_diagonal(X):
    X_ = X.copy()
    n = X.shape[0]

    for i in range(-15, 15):
        x = range(n)
        y = [x_ + i for x_ in x]
        
        if i != 0:
            x = x[abs(i):-abs(i)]
            y = y[abs(i):-abs(i)]
        X_[x,y] = 0
    return X_

    
def convolve_array(X, cfilter=scharr):
    grad = signal.convolve2d(X, cfilter, boundary='symm', mode='same')
    X_conv = np.absolute(grad)
    return X_conv


def convolve_array_tile(X, cfilter=scharr, divisor=49):
    """
    Iteratively convolve equal sized tiles in X, rejoining for fast convolution of the whole
    """
    x_height, x_width = X.shape

    assert x_height == x_width, "convolve_array expects square matrix"

    # Find even split for array
    divisor = divisor
    tile_height = None
    while (not tile_height) or (int(tile_height) != tile_height):
        # iterate divisor until whole number is found
        divisor += 1
        tile_height = x_height / divisor

    tile_height = int(tile_height)

    # Get list of tiles 
    tiled_array = X.reshape(divisor, tile_height, -1, tile_height)\
                   .swapaxes(1, 2)\
                   .reshape(-1, tile_height, tile_height)

    # Convolve tiles iteratively
    tiled_array_conv = np.array([convolve_array(x, cfilter=cfilter) for x in tiled_array])

    # Reconstruct original array using convolved tiles
    X_conv = tiled_array_conv.reshape(divisor, divisor, tile_height, tile_height)\
                             .swapaxes(1, 2)\
                             .reshape(x_height, x_width)

    return X_conv


def binarize(X, bin_thresh, filename=None):
    X_bin = X.copy()
    X_bin[X_bin < bin_thresh] = 0
    X_bin[X_bin >= bin_thresh] = 1

    if filename:
        skimage.io.imsave(filename, X_bin)

    return X_bin


def diagonal_gaussian(X, gauss_sigma, filename=False):
    d = X.shape[0]
    X_gauss = X.copy()

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1,d):
        diy = np.append(diag_indices_y, diag_indices_y[:i])
        diy = diy[i:]
        X_gauss[diag_indices_x, diy] = gaussian_filter(X_gauss[diag_indices_x, diy], sigma=gauss_sigma)

    diag_indices_x, diag_indices_y = np.diag_indices_from(X_gauss)
    for i in range(1,d):
        dix = np.append(diag_indices_x, diag_indices_x[:i])
        dix = dix[i:]
        X_gauss[dix, diag_indices_y] = gaussian_filter(X_gauss[dix, diag_indices_y], sigma=gauss_sigma)

    if filename:
        skimage.io.imsave(filename, X_gauss)

    return X_gauss


def plot_hough(image, h, theta, d, peaks, out_file):
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')

    ax[2].imshow(image, cmap=cm.gray)
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*peaks):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()


def hough_transform(X, min_dist_sec, cqt_window, hough_high_angle, hough_low_angle, hough_threshold, filename=None):
    # TODO: fix this
    hough_min_dist = int(min_dist_sec * cqt_window)

    if hough_high_angle == hough_low_angle:
        tested_angles = np.array([-hough_high_angle * np.pi / 180])
    else:
        tested_angles = np.linspace(- hough_low_angle * np.pi / 180, -hough_high_angle-1 * np.pi / 180, 100, endpoint=False) #np.array([-x*np.pi/180 for x in range(43,47)])

    h, theta, d = hough_line(X, theta=tested_angles)
    peaks = hough_line_peaks(h, theta, d, min_distance=hough_min_dist, min_angle=0, threshold=hough_threshold)
    
    if filename:
        plot_hough(X, h, theta, d, peaks, filename)

    return peaks


def hough_transform_new(X, hough_high_angle, hough_low_angle, hough_threshold, filename=None):

    lines = cv2.HoughLines(X.astype(np.uint8),1, np.pi/180, hough_threshold)
    lines = lines[:,0]
    
    upper_angle = ((90 + hough_high_angle) * np.pi / 180)
    lower_angle = ((90 + hough_low_angle) * np.pi / 180)
    
    lines = np.array([[dist, angle] for dist, angle in lines if lower_angle < angle < upper_angle])
    
    peaks = (list(range(len(lines[:,1]))), lines[:,1], lines[:,0])
    
    if filename:
        plot_hough_new(X, peaks, filename)

    return peaks


def plot_hough_new(X, peaks, out_file):
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.imshow(X, cmap=cm.gray)
    ax.set_ylim((X.shape[0], 0))
    ax.set_axis_off()
    ax.set_title('Detected lines')

    for _, angle, dist in zip(*peaks):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax.axline((x0, y0), slope=np.tan(angle + np.pi/2), linewidth=0.5, color='red', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_file)
    plt.clf()