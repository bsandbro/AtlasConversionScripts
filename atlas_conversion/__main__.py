import argparse
import errno
import os
import sys

from atlas_conversion.convertPNG import ImageSlices2TiledImage, load_png, write_versions, listdir_fullpath


######################################
# Main program - CLI with argparse - #
######################################
def main():
    # Define th CLI
    parser = argparse.ArgumentParser(prog='PNG Atlas Generator',
                                     description='''
PNG Atlas generation utility
----------------------------\n

This application converts the slices found in a folder into a tiled 2D texture
image in PNG format.\nIt uses Python with PIL, numpy and pydicom packages are recommended for other formats.
\n
Note: this version does not process several folders recursively.''',
                                     epilog='''
This code was created by Luis Kabongo.
Modified by Ander Arbelaiz to add gradient calculation.\n
Information links:
 - https://github.com/VolumeRC/AtlasConversionScripts/wiki
 - http://www.volumerc.org
 - http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org''',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='must contain a path to one set of PNG files to be processed')
    parser.add_argument('output', type=str,
                        help='must contain the path and base name of the desired output,\n'
                             'extension will be added automatically')
    parser.add_argument('--resize', '-r', type=int, nargs=2, metavar=('x', 'y'),
                        help='resizing of the input images x y, before processing')
    parser.add_argument('--gradient', '-g', action='store_true',
                        help='calculate and generate the gradient atlas')
    parser.add_argument('--standard_deviation', '-std', type=int, default=2,
                        help='standard deviation for the gaussian kernel used for the gradient computation')

    # Obtain the parsed arguments
    print("Parsing arguments...")
    arguments = parser.parse_args()

    # Filter only png files in the given folder
    filenames_png = [x for x in listdir_fullpath(arguments.input) if ".png" in x]

    if not len(filenames_png) > 0:
        print("No PNG files found in that folder, check your parameters or contact the authors :).")
        return 2

    if arguments.resize:
        width, height = arguments.resize[0], arguments.resize[1]
    else:
        width, height = None, None

    # Update global value for standard_deviation
    sigmaValue = arguments.standard_deviation

    c_gradient = False
    if arguments.gradient:
        try:
            global ndimage, misc, np, da, delayed, h5py
            import numpy as np
            import dask.array as da
            import h5py
            from dask import delayed
            from scipy import ndimage, misc
            c_gradient = True
        except ImportError:
            print("You need the following dependencies to also calculate the gradient: scipy, numpy, h5py and dask")

    # From png files
    imgTile, gradientTile, sliceResolution, numberOfSlices, slicesPerAxis = ImageSlices2TiledImage(filenames_png,
                                                                                                   load_png,
                                                                                                   c_gradient,
                                                                                                   width,
                                                                                                   height)

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
            "containing dimensions (total slices, slices per axis):", (numberOfSlices, (slicesPerAxis, slicesPerAxis)))
    else:
        print("Created", arguments.output + "_AtlasDim.txt", "containing dimensions (total slices, slices per axis):",\
            (numberOfSlices, (slicesPerAxis, slicesPerAxis)))

    # Output is written in different sizes
    write_versions(imgTile, gradientTile, arguments.output)

if __name__ == "__main__":
    sys.exit(main())