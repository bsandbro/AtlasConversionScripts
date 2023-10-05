import os
import numpy as np
from PIL import Image
from dask import delayed
import dask.array as da
from scipy import ndimage


# Simple decrement function
def decr(x, y):
    return x - y


# Normalize values between [0-1]
def normalize(block):
    old_min = delayed(block.min())
    old_max = delayed(block.max())
    r = delayed(decr)(old_max, old_min)
    minimum = old_min.compute()
    t0 = decr(block, minimum)
    return t0 / r.compute(), -minimum / r.compute()


# Calculate derivatives function
def gaussian_filter(block, axis, sigma_value=2):
    return ndimage.gaussian_filter1d(block, sigma=sigma_value, axis=axis, order=1)


# This function calculates the gradient from a 3-dimensional dask array
def calculate_gradient(slices):
    axes = [1, 0, 2]
    overlap_depth = {0: 1, 1: 1, 2: 1}
    overlap_boundary = {0: 'reflect', 1: 'reflect', 2: 'reflect'}
    g = da.overlap.overlap(slices, depth=overlap_depth, boundary=overlap_boundary)
    derivatives = [g.map_blocks(gaussian_filter, axis) for axis in axes]
    derivatives = [da.overlap.trim_overlap(d, depth=overlap_depth,  boundary=overlap_boundary) for d in derivatives]
    gradient = da.stack(derivatives, axis=3).compute()
    return normalize(gradient)


# This functions takes a (tiled) image and writes it to a png file with base filename outputFilename.
# It also writes several versions in different sizes determined by dimensions
def write_versions(tileImage, tileGradient, outputFilename, dimensions=None):
    if dimensions is None:
        dimensions = [8192, 4096, 2048, 1024, 512, 256]
    tileImage.save(outputFilename + "_full.png", "PNG")
    if tileGradient:
        tileGradient.save(outputFilename + "_gradient_full.png", "PNG")
    for dimension in [x for x in dimensions if x < tileImage.size[0]]:
        print("Writing image with size: " + str(dimension) + "...")
        im = tileImage.resize((dimension, dimension), Image.BICUBIC)
        im.save(outputFilename + "_" + str(dimension) + ".png")
        if tileGradient:
            im = tileGradient.resize((dimension, dimension), Image.BICUBIC)
            print("Writing gradient with size: " + str(dimension) + "...")
            im.save(outputFilename + "_" + str(dimension) + "_gradient.png")


# This function lists the files within a given directory dir
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
