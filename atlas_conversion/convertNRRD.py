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

import os, errno
import sys
import math
import argparse
from argparse import RawTextHelpFormatter
from multiprocessing import cpu_count
import tempfile
from PIL import Image
import numpy as np
from scipy import ndimage, misc

from atlas_conversion.utils import calculate_gradient


# This function simply loads a NRRD file and returns a compatible Image object
def loadNRRD(filename):
    try:
        data, info = nrrd.read(filename)
        return data
    except:
        print('Error reading the nrrd file!')
        exit()


# This function uses the images retrieved with loadImgFunction (whould return a PIL.Image) and
#	writes them as tiles within a new square Image. 
#	Returns a set of Image, size of a slice, number of slices and number of slices per axis
def ImageSlices2TiledImage(filename, loadImgFunction=loadNRRD, cGradient=False):
    print("Desired load function=", loadImgFunction.__name__)
    data = loadImgFunction(filename)
    volumeSize = (data.shape[0], data.shape[1])
    numberOfSlices = data.shape[2]
    slicesPerAxis = int(math.ceil(math.sqrt(numberOfSlices)))
    atlasArray = np.zeros((volumeSize[0] * slicesPerAxis, volumeSize[1] * slicesPerAxis))

    for i in range(0, numberOfSlices):
        row = int((math.floor(i / slicesPerAxis)) * volumeSize[0])
        col = int((i % slicesPerAxis) * volumeSize[1])
        box = (row, col, int(row + volumeSize[0]), int(col + volumeSize[1]))
        atlasArray[box[0]:box[2], box[1]:box[3]] = data[:, :, i]

    # From numpy to PIL image
    imout = misc.toimage(atlasArray, mode="L")

    gradient = None
    if cGradient:
        print("Starting to compute the gradient: Loading the data...")
        cpus = cpu_count()
        chunk_size = [x // cpus for x in data.shape]
        print("Calculated chunk size: " + str(chunk_size))
        data = da.from_array(data, chunks=chunk_size)
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
                             (volumeSize[0] * slicesPerAxis, volumeSize[1] * slicesPerAxis),
                             (g_background, g_background, g_background))

        channels = ['/r', '/g', '/b']
        handle = h5py.File(f.name)
        dsets = [handle[c] for c in channels]
        arrays = [da.from_array(dset, chunks=chunk_size) for dset in dsets]
        gradient_data = da.stack(arrays, axis=-1)

        for i in range(0, numberOfSlices):
            row = int((math.floor(i / slicesPerAxis)) * volumeSize[0])
            col = int((i % slicesPerAxis) * volumeSize[1])
            box = (int(col), int(row), int(col + volumeSize[0]), int(row + volumeSize[1]))

            s = gradient_data[:, :, i, :]
            im = Image.fromarray(np.array(s))
            gradient.paste(im, box)
        try:
            handle.close()
            f.close()
        finally:
            try:
                os.remove(f.name)
            except OSError as e:  # this would be "except OSError, e:" before Python 2.6
                if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
                    raise  # re-raise exception if a different error occurred
    return imout, gradient, volumeSize, numberOfSlices, slicesPerAxis


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


# This is the main program, it takes at least 2 arguments <InputFolder> and <OutputFilename>
def main(argv=None):
    # Define th CLI
    parser = argparse.ArgumentParser(prog='NRRD Atlas Generator',
                                     description='''
NRRD Atlas generation utility
-----------------------------\n

This application converts the slices found in a folder into a tiled 2D texture
image in PNG format.\nIt uses Python with PIL, numpy and pydicom packages are recommended for other formats.
''',
                                 epilog='''
This code was created by Luis Kabongo.
Modified by Ander Arbelaiz to add gradient calculation.\n
Information links:
 - https://github.com/VolumeRC/AtlasConversionScripts/wiki
 - http://www.volumerc.org
 - http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org''',
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='must contain a path to the NRRD file to be processed')
    parser.add_argument('output', type=str,
                        help='must contain the path and base name of the desired output,\n'
                             'extension will be added automatically')
    parser.add_argument('--gradient', '-g', action='store_true',
                        help='calculate and generate the gradient atlas')
    parser.add_argument('--standard_deviation', '-std', type=int, default=2,
                        help='standard deviation for the gaussian kernel used for the gradient computation')

    # Obtain the parsed arguments
    print("Parsing arguments...")
    arguments = parser.parse_args()

    filenameNRRD = arguments.input

    if not len(filenameNRRD) > 0:
        print("No NRRD file found in that folder, check your parameters or contact the authors :).")
        return 2

    # Convert into a tiled image
    try:
        global nrrd
        import nrrd
    except:
        print("You need pynrrd package (sudo easy_install pynrrd) to do this!")
        return 2

    # Update global value for standard_deviation
    sigmaValue = arguments.standard_deviation

    c_gradient = False
    if arguments.gradient:
        try:
            global da, delayed, h5py
            import dask.array as da
            import h5py
            from dask import delayed
            c_gradient = True
        except ImportError:
            print("You need the following dependencies to also calculate the gradient: numpy, h5py and dask")

    # From nrrd files
    imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenameNRRD,
                                                                                                   loadNRRD,
                                                                                                   c_gradient)

    # Write a text file containing the number of slices for reference
    try:
        try:
            print('Creating folder', os.path.dirname(arguments.output), '...', end=' ')
            os.makedirs(os.path.dirname(arguments.output))
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(os.path.dirname(arguments.output)):
                print('was already there.')
            else:
                print(', folders might not be created, trying to write anyways...')
        except:
            print(", could not create folders, trying to write anyways...")
        with open(str(arguments.output) + "_AtlasDim.txt", 'w') as f:
            f.write(str((numberOfSlices, (slicesPerAxis, slicesPerAxis))))
    except:
        print("Could not write a text file", str(arguments.output) + "_AtlasDim.txt", \
            "containing dimensions (total slices, slices per axis):", (
        numberOfSlices, (slicesPerAxis, slicesPerAxis)))
    else:
        print("Created", arguments.output + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):", \
            (numberOfSlices, (slicesPerAxis, slicesPerAxis)))

    # Output is written in different sizes
    write_versions(imgTile, gradientTile, arguments.output)


if __name__ == "__main__":
    sys.exit(main())
