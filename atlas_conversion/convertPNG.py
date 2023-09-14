#!/usr/bin/env python
# coding=utf-8
#
# Copyright 2012 Vicomtech-IK4
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import errno
import math
from multiprocessing import cpu_count
import tempfile
import dask.array as da
import h5py
import numpy as np
from PIL import Image

from atlas_conversion.utils import calculate_gradient

# This is the default size when loading a Raw image
sizeOfRaw = (512, 512)
# This determines if the endianness should be reversed
rawByteSwap = True

# This function simply loads a PNG file and returns a compatible Image object
def load_png(filename):
    im = Image.open(filename)
    if im.mode != 1:
        return im.convert("L", palette=Image.ADAPTIVE, colors=256)
    return im


def resize_image(im, width, height, _filter=Image.BICUBIC):
    if width is not None or height is not None:
        original_size = im.size
        if width is None:
            width = original_size[0]
        if height is None:
            height = original_size[1]
        size = (width, height)
        return im.resize(size, _filter)
    return im


def make_square_image(im):
    mode = im.mode
    width, height = im.size
    new_background = 0  # L, 1
    if len(mode) == 3:  # RGB
        new_background = (0, 0, 0)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (0, 0, 0, 0)
    new_resolution = max(width, height)
    offset = ((new_resolution - width) / 2, (new_resolution - height) / 2)
    t_im = Image.new("L", (new_resolution, new_resolution), new_background)
    t_im.paste(im, offset)
    return t_im


def read_image(filename, load_img_func=load_png, r_width=None, r_height=None):
    # Load the image
    im = load_img_func(filename)
    # Perform resize if required
    im = resize_image(im, r_width, r_height)
    # Create an square image if required
    width, height = im.size
    if width != height:
        return make_square_image(im)
    return im


# This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
# writes them as tiles within a new square Image.
# Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filenames, loadImgFunction=load_png, cGradient=False, r_width=None, r_height=None):
    filenames = sorted(filenames)
    print(("Desired load function=", loadImgFunction.__name__))
    size = read_image(filenames[0], loadImgFunction, r_width, r_height).size
    numberOfSlices = len(filenames)
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))
    imout = Image.new("L", (size[0] * slicesPerAxis, size[1] * slicesPerAxis))

    i = 0
    for filename in filenames:
        im = read_image(filename, loadImgFunction, r_width, r_height)

        row = int((math.floor(i / slicesPerAxis)) * size[0])
        col = int((i % slicesPerAxis) * size[1])

        box = (int(col), int(row), int(col + size[0]), int(row + size[1]))
        imout.paste(im, box)

        i += 1
        print(("processed slice  : " + str(i) + "/" + str(numberOfSlices)))  # filename

    gradient = None
    if cGradient:
        print("Starting to compute the gradient: Loading the data...")
        image_list = [da.from_array(np.array(read_image(f, loadImgFunction, r_width, r_height),
                                             dtype='uint8'), chunks=size) for f in filenames]
        data = da.stack(image_list, axis=-1)
        data = da.rechunk(data)
        print(("Loading complete. Data size: "+str(data.shape)))
        print("Computing the gradient...")
        data = data.astype(np.float32)
        gradient_data, g_background = calculate_gradient(data)
        # Normalize values to RGB values
        gradient_data *= 255
        g_background = int(g_background * 255)
        gradient_data = gradient_data.astype(np.uint8)
        # Keep the RGB information separated, uses less RAM memory
        channels = ['/r', '/g', '/b']
        f = tempfile.NamedTemporaryFile(delete=False)
        [da.to_hdf5(f.name, c, gradient_data[:, :, :, i]) for i, c in enumerate(channels)]
        print("Computed gradient data saved in cache file.")
        # Create atlas image
        gradient = Image.new("RGB",
                             (size[0] * slicesPerAxis, size[1] * slicesPerAxis),
                             (g_background, g_background, g_background))

        channels = ['/r', '/g', '/b']
        handle = h5py.File(f.name)
        dsets = [handle[c] for c in channels]
        arrays = [da.from_array(dset) for dset in dsets]
        gradient_data = da.stack(arrays, axis=-1)

        for i in range(0, numberOfSlices):
            row = int((math.floor(i / slicesPerAxis)) * size[0])
            col = int((i % slicesPerAxis) * size[1])
            box = (int(col), int(row))

            s = gradient_data[:, :, i, :]
            im = Image.fromarray(np.array(s))
            gradient.paste(im,box)
            print(("processed gradient slice  : " + str(i+1) + "/" + str(numberOfSlices)))  # filename

        try:
            handle.close()
            f.close()
        finally:
            try:
                os.remove(f.name)
            except OSError as e:  # this would be "except OSError, e:" before Python 2.6
                if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                    raise  # re-raise exception if a different error occurred
    return imout, gradient, size, numberOfSlices, slicesPerAxis




