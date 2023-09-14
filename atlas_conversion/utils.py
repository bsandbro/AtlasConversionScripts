from atlas_conversion.convertPNG import sigmaValue
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
def gaussian_filter(block, axis, sigmaValue=2):
    return ndimage.gaussian_filter1d(block, sigma=sigmaValue, axis=axis, order=1)


# This function calculates the gradient from a 3 dimensional dask array
def calculate_gradient(arr):
    axises = [1, 0, 2]  # Match RGB
    g = da.overlap.overlap(arr, depth={0: 1, 1: 1, 2: 1}, boundary={0: 'reflect', 1: 'reflect', 2: 'reflect'})
    derivatives = [g.map_blocks(gaussian_filter, axis) for axis in axises]
    derivatives = [da.overlap.trim_internal(d, {0: 1, 1: 1, 2: 1}) for d in derivatives]
    gradient = da.stack(derivatives, axis=3)
    return normalize(gradient)
