import argparse
import sys

from atlas_conversion.atlas import Atlas
from atlas_conversion.loaders import png_loader, dicom_loader, nrrd_loader, raw_loader


######################################
# Main program - CLI with argparse - #
######################################
def main():
    # Define th CLI
    parser = argparse.ArgumentParser(prog='Atlas Generator',
                                     description='''
Atlas generation utility
--------------------------\n

This application converts the slices found in a folder into a tiled 2D texture
image in PNG format.\nIt uses Python with PIL, numpy and pydicom packages.
\n
Note: this version does not process several folders recursively.''',
                                     epilog='''
This code was created by Luis Kabongo.
Modified by Ander Arbelaiz to add gradient calculation.\n
Modified by Ben Sandbrook for python 3 compatibility, packaging and CLI.\n
Information links:
 - https://github.com/VolumeRC/AtlasConversionScripts/wiki
 - http://www.volumerc.org
 - http://demos.vicomtech.org
Contact mailto:volumerendering@vicomtech.org''',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', type=str, help='path to the set of files to be processed')
    parser.add_argument('output', type=str,
                        help='path and base name of the desired output file, extension added automatically')
    parser.add_argument('--resize', '-r', type=int, nargs=2, metavar=('x', 'y'),
                        help='resize the input images x y before processing')
    parser.add_argument('--gradient', '-g', action='store_true',
                        help='calculate and generate the gradient atlas')
    parser.add_argument('--standard_deviation', '-std', type=int, default=2,
                        help='standard deviation for the gaussian kernel used for the gradient computation')
    parser.add_argument('--format', '-f', type=str, default="png", choices=["png", "dicom", "nrrd", "raw"],
                        help='format of the input images, default is png')

    # Obtain the parsed arguments
    print("Parsing arguments...")
    arguments = parser.parse_args()

    loaders_map = {
        "png": (png_loader, ".png"),
        "dicom": (dicom_loader, ".dcm"),
        "nrrd": (nrrd_loader, ".nrrd"),
        "raw": (raw_loader, ".raw")
    }

    loader, ext = loaders_map[arguments.format]

    print("Loading images...")
    atlas_obj = Atlas(loader, resize=arguments.resize)
    atlas_obj.load(arguments.input)

    print("Converting images...")
    atlas_obj.convert()

    print("Writing images...")
    atlas_obj.write(arguments.output, gradient=arguments.gradient)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
