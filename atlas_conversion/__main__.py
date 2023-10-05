import argparse
import sys

from atlas_conversion.atlas import Atlas
from atlas_conversion.loaders import png_loader, dicom_loader, nrrd_loader, raw_loader


######################################
# Main program - CLI with argparse - #
######################################
def main():
    parser = create_parser()
    arguments = check_and_parse_args(parser)

    loaders_map = {
        "png": png_loader,
        "dicom": dicom_loader,
        "nrrd": nrrd_loader,
        "raw": raw_loader
    }

    loader = loaders_map[arguments.format]

    print("Loading images...")
    atlas_obj = Atlas(loader, resize=arguments.resize, **arguments.loader_options)
    atlas_obj.load(arguments.input)

    print("Converting images...")
    atlas_obj.convert()

    if arguments.gradient:
        print("Calculating gradient and writing images... (this may take a while)")
    else:
        print("Writing images...")
    atlas_obj.write(arguments.output, gradient=arguments.gradient)

    print("Done!")
    return 0


def create_parser():
    parser = argparse.ArgumentParser(prog='Atlas Generator',
                                     description='''
Atlas generation utility
--------------------------

This application converts the slices found in a folder into a tiled 2D texture
image in PNG format.
It uses Python 3 with Pillow, numpy, pydicom, nrrd, scipy, and dask.

Note: this version does not process several folders recursively.''',
                                     epilog='''
This code was created by Luis Kabongo.
Modified by Ander Arbelaiz to add gradient calculation.
Modified by Ben Sandbrook for python 3 compatibility, packaging, CLI, and testing.
Information links:
 - https://github.com/VolumeRC/AtlasConversionScripts/wiki
 - http://www.volumerc.org
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
    parser.add_argument('--raw-size', nargs=2, metavar=('width', 'height'), type=int,
                        help='Size of the raw image, required if format is raw, otherwise ignored.')
    parser.add_argument('--raw-slices', type=int,
                        help='Number of slices in the raw data, required if format is raw, otherwise ignored..')
    parser.add_argument('--raw-channels', type=int, choices=[1, 3],
                        help='Number of channels (1 for grayscale, 3 for RGB) in the raw data, required if format is '
                             'raw, otherwise ignored.')

    return parser


def check_and_parse_args(parser):
    print("Parsing arguments...")
    arguments = parser.parse_args()
    loader_options = {}
    if arguments.resize:
        loader_options["resize"] = arguments.resize
    if arguments.format == "raw":
        check_raw_format_requirements(arguments, parser)
        loader_options["size"] = arguments.raw_size
        loader_options["num_slices"] = arguments.raw_slices
        loader_options["num_channels"] = arguments.raw_channels
    arguments.loader_options = loader_options
    return arguments


def check_raw_format_requirements(arguments, parser):
    """Ensure all required arguments for raw format are present."""
    required_args = {
        "raw_size": "--raw-size is required if format is raw",
        "raw_slices": "--raw-slices is required if format is raw",
        "raw_channels": "--raw-channels is required if format is raw"
    }
    for arg, error_msg in required_args.items():
        if getattr(arguments, arg) is None:
            parser.error(error_msg)

if __name__ == "__main__":
    sys.exit(main())
