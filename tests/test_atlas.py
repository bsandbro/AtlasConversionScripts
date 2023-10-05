import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from atlas_conversion.atlas import Atlas


def dummy_loader(path=None, size=(128, 128), num_slices=10):
    images = []
    for i in range(num_slices):
        increment = int(255 * (i + 1) / num_slices)
        image = np.full((*size, 3), increment, dtype=np.uint8)
        images.append(image)
    return images


class TestAtlasClass(unittest.TestCase):
    def test_initialization(self):
        atlas = Atlas(dummy_loader)
        self.assertIsNotNone(atlas)
        self.assertEqual(atlas.loader, dummy_loader)
        self.assertIsNone(atlas.atlas)
        self.assertIsNone(atlas.slices)

    def test_load(self):
        atlas = Atlas(dummy_loader)
        atlas.load("")
        self.assertIsNotNone(atlas.slices)
        self.assertEqual(len(atlas.slices), 10)
        for i, image in enumerate(atlas.slices):
            increment = int(255 * (i + 1) / 10)
            self.assertTrue(np.all(image == increment))


class TestAtlasConversion(unittest.TestCase):
    def setUp(self):
        # Create an Atlas instance using the dummy loader
        self.atlas_obj = Atlas(dummy_loader)
        self.atlas_obj.load("")  # Load dummy images
        self.atlas_obj.convert()  # Convert into an atlas

    def test_convert(self):
        # Check if the atlas is created
        self.assertIsNotNone(self.atlas_obj.atlas)

        # Since there are 10 images, the size should be 4 (ceil(sqrt(10))).
        self.assertEqual(self.atlas_obj.size, 4)

        # Check atlas shape
        self.assertEqual(self.atlas_obj.atlas.shape, (512, 512, 3))  # 4 images * 128 pixels each

        # Check if the images are placed correctly with increasing color values
        for i in range(10):
            increment = int(255 * (i + 1) / 10)
            row = (i // self.atlas_obj.size) * 128
            col = (i % self.atlas_obj.size) * 128
            # Extract the corresponding slice from atlas
            slice_from_atlas = self.atlas_obj.atlas[row:row+128, col:col+128, :]
            # Check if the color is uniform and matches the expected increment
            self.assertTrue(np.all(slice_from_atlas == increment))


class TestGradientComputation(unittest.TestCase):

    def setUp(self):
        self.atlas_obj = Atlas(dummy_loader)
        self.atlas_obj.load("")
        self.atlas_obj.convert()

    def test_compute_gradient_basic_z(self):
        gradient_image = self.atlas_obj.compute_gradient()

        # Convert the PIL image to a numpy array
        gradient_data = np.array(gradient_image)

        # Check the gradient image type and shape
        self.assertIsInstance(gradient_image, Image.Image)
        self.assertEqual(gradient_image.size, (512, 512))  # 4 images * 128 pixels each in a square format
        self.assertEqual(gradient_data.shape, (512, 512, 3))  # 3 channels (RGB)

        for i in range(3):
            for j in range(3):
                # Extracting a 128x128 block which corresponds to a slice
                block = gradient_data[128 * i:128 * (i + 1), 128 * j:128 * (j + 1)]

                if i * 4 + j < 10:
                    # Since there's no variation in the X and Y directions, R and G channels should be 0
                    self.assertTrue(np.all(block[:, :, 0] == 0))
                    self.assertTrue(np.all(block[:, :, 1] == 0))
                    # Check the gradient in the Z-direction is non-zero
                    self.assertTrue(np.all(block[:, :, 2] != 0))


class TestAtlasFileOutput(unittest.TestCase):

    def setUp(self):
        self.atlas_obj = Atlas(dummy_loader, size=(512, 512), num_slices=512)
        self.atlas_obj.load("")
        self.atlas_obj.convert()

    def test_write_with_gradient(self):
        with TemporaryDirectory() as tmpdirname:
            output_base_path = os.path.join(tmpdirname, "test_output")
            self.atlas_obj.write(output_base_path, gradient=True)

            # Now, check if files have been created
            self.assertTrue(os.path.exists(output_base_path + "_AtlasDim.txt"))
            self.assertTrue(os.path.exists(output_base_path + "_full.png"))
            self.assertTrue(os.path.exists(output_base_path + "_gradient_full.png"))

            # Check for various sizes
            for size in [8192, 4096, 2048, 1024, 512]:
                if size < self.atlas_obj.atlas.shape[0]:
                    self.assertTrue(os.path.exists(f"{output_base_path}_{size}.png"))
                    self.assertTrue(os.path.exists(f"{output_base_path}_{size}_gradient.png"))

    def test_write(self):
        with TemporaryDirectory() as tmpdirname:
            output_base_path = os.path.join(tmpdirname, "test_output")
            self.atlas_obj.write(output_base_path, gradient=False)

            # Now, check if files have been created
            self.assertTrue(os.path.exists(output_base_path + "_AtlasDim.txt"))
            self.assertTrue(os.path.exists(output_base_path + "_full.png"))

            # Check for various sizes
            for size in [8192, 4096, 2048, 1024, 512]:
                if size < self.atlas_obj.atlas.shape[0]:
                    self.assertTrue(os.path.exists(f"{output_base_path}_{size}.png"))

