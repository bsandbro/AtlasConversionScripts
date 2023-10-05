import unittest
import os
import random
import nrrd

from atlas_conversion.loaders import png_loader, dicom_loader, nrrd_loader, raw_loader
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from PIL import Image
import numpy as np
from tempfile import TemporaryDirectory


def generate_random_image(size=(128, 128)):
    """Generate a random RGB image of the specified size."""
    return Image.fromarray(
        np.array(
            [[[random.randint(0, 255) for _ in range(3)] for _ in range(size[0])] for _ in range(size[1])],
            dtype=np.uint8
        )
    )


class TestDICOM(unittest.TestCase):

    def setUp(self):
        self.num_slices = 10
        # Generate random slices
        self.slices = [generate_random_image() for _ in range(self.num_slices)]

    def generate_dicom_slices(self, temp_dir):
        """Generate minimal DICOM slices."""
        for idx, slice_img in enumerate(self.slices):
            ds = Dataset()

            file_meta = FileMetaDataset()
            file_meta.MediaStorageSOPInstanceUID = generate_uid()
            file_meta.MediaStorageSOPClassUID = generate_uid()
            file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.file_meta = file_meta
            ds.SOPInstanceUID = generate_uid()
            ds.SeriesInstanceUID = generate_uid()
            ds.StudyInstanceUID = generate_uid()
            ds.SliceLocation = idx
            ds.Rows, ds.Columns = slice_img.size
            ds.PixelData = slice_img.tobytes()
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.SamplesPerPixel = 3
            ds.PhotometricInterpretation = "RGB"
            ds.PlanarConfiguration = 0
            ds.is_little_endian = True
            ds.is_implicit_VR = False

            ds.save_as(os.path.join(temp_dir, f"slice_{idx}.dcm"))

    def test_dicom_loader(self):
        with TemporaryDirectory() as temp_dir:
            self.generate_dicom_slices(temp_dir)

            loaded_slices = dicom_loader(temp_dir)
            self.assertEqual(len(loaded_slices), self.num_slices)
            for loaded, original in zip(loaded_slices, self.slices):
                self.assertTrue(np.array_equal(loaded, np.array(original)))


class TestPNG(unittest.TestCase):

    def setUp(self):
        self.num_slices = 10
        # Generate random slices
        self.slices = [generate_random_image() for _ in range(self.num_slices)]

    def generate_png_slices(self, temp_dir):
        """Generate random PNG slices."""
        for idx, slice_img in enumerate(self.slices):
            slice_img.save(os.path.join(temp_dir, f"slice_{idx}.png"))

    def test_png_loader(self):
        with TemporaryDirectory() as temp_dir:
            self.generate_png_slices(temp_dir)

            loaded_slices = png_loader(temp_dir)
            self.assertEqual(len(loaded_slices), self.num_slices)
            for loaded, original in zip(loaded_slices, self.slices):
                self.assertTrue(np.array_equal(loaded, np.array(original)))


class TestNRRD(unittest.TestCase):

    def setUp(self):
        self.num_slices = 10
        # Generate random slices and stack them into a volume
        self.slices = [generate_random_image() for _ in range(self.num_slices)]
        self.volume = np.stack([np.array(slice_) for slice_ in self.slices], axis=2)

    def generate_nrrd_volume(self, filepath):
        """Save the generated volume as an NRRD file."""
        nrrd.write(filepath, self.volume)

    def test_nrrd_loader(self):
        with TemporaryDirectory() as temp_dir:
            nrrd_path = os.path.join(temp_dir, "volume.nrrd")
            self.generate_nrrd_volume(nrrd_path)

            loaded_slices = nrrd_loader(nrrd_path)
            self.assertEqual(len(loaded_slices), self.num_slices)

            for loaded_slice, original_slice in zip(loaded_slices, self.slices):
                self.assertTrue(np.array_equal(loaded_slice, np.array(original_slice)))


class TestRAW(unittest.TestCase):

    def setUp(self):
        self.num_slices = 10
        # Generate random slices
        self.slices = [generate_random_image() for _ in range(self.num_slices)]

    def generate_raw_data_rgb(self, filepath):
        """Save the generated slices as a raw binary file."""
        with open(filepath, 'wb') as raw_file:
            for slice_ in self.slices:
                raw_file.write(slice_.tobytes())

    def generate_raw_data_grayscale(self, filepath):
        """Save the generated slices as a raw binary file."""
        with open(filepath, 'wb') as raw_file:
            for slice_ in self.slices:
                raw_file.write(slice_.convert('L').tobytes())

    def test_raw_loader_rgb(self):
        with TemporaryDirectory() as temp_dir:
            raw_path = os.path.join(temp_dir, "volume.raw")
            self.generate_raw_data_rgb(raw_path)

            loaded_slices = raw_loader(raw_path, size_of_raw=(128, 128), slices=self.num_slices, channels=3)
            self.assertEqual(len(loaded_slices), self.num_slices)

            for loaded_slice, original_slice in zip(loaded_slices, self.slices):
                self.assertTrue(np.array_equal(loaded_slice, np.array(original_slice)))

    def test_raw_loader_generate_raw_data_grayscale(self):
        with TemporaryDirectory() as temp_dir:
            raw_path = os.path.join(temp_dir, "volume.raw")
            self.generate_raw_data_grayscale(raw_path)

            loaded_slices = raw_loader(raw_path, size_of_raw=(128, 128), slices=self.num_slices, channels=1)
            self.assertEqual(len(loaded_slices), self.num_slices)

            for loaded_slice, original_slice in zip(loaded_slices, self.slices):
                grayscale_original_slice_array = np.array(original_slice.convert('L'))
                self.assertTrue(np.array_equal(loaded_slice, grayscale_original_slice_array))

    def test_raw_fails_without_size_of_raw(self):
        with TemporaryDirectory() as temp_dir:
            raw_path = os.path.join(temp_dir, "volume.raw")
            self.generate_raw_data_rgb(raw_path)

            with self.assertRaises(TypeError):
                raw_loader(raw_path, size_of_raw=None, slices=self.num_slices, channels=3)
