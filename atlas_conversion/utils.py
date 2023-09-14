import os
import errno
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


# This function calculates the gradient from a 3 dimensional dask array
def calculate_gradient(arr):
    axes = [1, 0, 2]  # Match RGB
    g = da.overlap.overlap(arr, depth={0: 1, 1: 1, 2: 1}, boundary={0: 'reflect', 1: 'reflect', 2: 'reflect'})
    derivatives = [g.map_blocks(gaussian_filter, axis) for axis in axes]
    derivatives = [da.overlap.trim_internal(d, {0: 1, 1: 1, 2: 1}) for d in derivatives]
    gradient = da.stack(derivatives, axis=3)
    return normalize(gradient)


# This functions takes a (tiled) image and writes it to a png file with base filename outputFilename.
# It also writes several versions in different sizes determined by dimensions
def write_versions(tileImage, tileGradient, outputFilename, dimensions=None):
    if dimensions is None:
        dimensions = [8192, 4096, 2048, 1024, 512]
    try:
        print('Creating folder', os.path.dirname(outputFilename), '...', end=' ')
        os.makedirs(os.path.dirname(outputFilename))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(outputFilename)):
            print('was already there.')
        else:
            print(', folders might not be created, trying to write anyways...')
    except:
        print("Could not create folders, trying to write anyways...")

    print("Writing complete image: " + outputFilename + "_full.png")
    try:
        tileImage.save(outputFilename + "_full.png", "PNG")
        if tileGradient:
            tileGradient.save(outputFilename + "_gradient_full.png", "PNG")
    except:
        print("Failed writing ", outputFilename + "_full.png")
    for dim in dimensions:
        if tileImage.size[0] > dim:
            print("Writing " + str(dim) + "x" + str(dim) + " version: " + outputFilename + "_" + str(dim) + ".png")
            try:
                tmpImage = tileImage.resize((dim, dim))
                tmpImage.save(outputFilename + "_" + str(dim) + ".png", "PNG")
            except:
                print("Failed writing ", outputFilename, "_", str(dim), ".png")
            if tileGradient:
                try:
                    tmpImage = tileGradient.resize((dim, dim))
                    tmpImage.save(outputFilename + "_gradient_" + str(dim) + ".png", "PNG")
                except:
                    print("Failed writing ", outputFilename, "_gradient_", str(dim), ".png")


# This function lists the files within a given directory dir
def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]
