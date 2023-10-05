import math
import dask.array as da
import numpy as np
from PIL import Image

from atlas_conversion.utils import normalize_rgb, calculate_gradient


class Atlas:
    """
    Atlas class

    This is a generic class for atlas conversion. It takes a loader function that loads the specific file types and
    returns them as RGB8 numpy arrays.

    Attributes:
        loader: a loader function that loads the atlas slices as list of RGB8 numpy arrays
    """

    def __init__(self, loader, **loader_options):
        self.atlas = None
        self.slices = None
        self.loader = loader
        self.size = None
        self.loader_options = loader_options

    def load(self, path):
        self.slices = self.loader(path, **self.loader_options)

    def convert(self):
        # create a square array large enough to hold all the slices
        self.size = int(math.ceil(math.sqrt(len(self.slices))))
        self.atlas = np.zeros((self.size * self.slices[0].shape[0], self.size * self.slices[0].shape[1], 3),
                              dtype=np.uint8)
        # copy the slices into the atlas taking into account the data shape
        for i, image_slice in enumerate(self.slices):
            row = int((math.floor(i / self.size)) * self.slices[0].shape[0])
            col = int((i % self.size) * self.slices[0].shape[1])
            row_dim, col_dim = self.slices[0].shape[0], self.slices[0].shape[1]
            self.atlas[row:row + row_dim, col:col + col_dim, :] = image_slice

    def write(self, output_filename, gradient=False):
        with open(str(output_filename) + "_AtlasDim.txt", 'w') as f:
            f.write(str((len(self.slices), (self.size, self.size))))
        im = Image.fromarray(self.atlas)
        im.save(output_filename + "_full.png", "PNG")
        if gradient:
            gradient = self.compute_gradient()
            gradient.save(output_filename + "_gradient_full.png", "PNG")
        for dimension in [x for x in [8192, 4096, 2048, 1024, 512] if x < self.atlas.shape[0]]:
            print("Writing image with size: " + str(dimension) + "...")
            im = im.resize((dimension, dimension), Image.BICUBIC)
            im.save(output_filename + "_" + str(dimension) + ".png")
            if gradient:
                print("Writing gradient with size: " + str(dimension) + "...")
                output_filename__resized_gradient = output_filename + "_" + str(dimension) + "_gradient.png"
                gradient.resize((dimension, dimension), Image.BICUBIC).save(output_filename__resized_gradient)

    def compute_gradient(self):
        slice_shape = self.slices[0].shape
        slices = [Image.fromarray(slice_).convert('L') for slice_ in self.slices]
        data = da.stack(slices, axis=-1).rechunk().astype(np.float32)
        gradient_data, g_background = calculate_gradient(data)
        g_background, gradient_data = normalize_rgb(g_background, gradient_data)

        atlas_array = np.full((slice_shape[0] * self.size, slice_shape[1] * self.size, 3), g_background)
        for i, _ in enumerate(self.slices):
            row = (i // self.size) * slice_shape[0]
            col = (i % self.size) * slice_shape[1]
            atlas_array[row:row + slice_shape[0], col:col + slice_shape[1], :] = gradient_data[:, :, i]

        return Image.fromarray(atlas_array.astype(np.uint8))
