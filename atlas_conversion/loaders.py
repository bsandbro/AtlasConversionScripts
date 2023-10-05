import os
import pydicom
import numpy as np
import nrrd
from PIL import Image


def resize_image(image, size, interpolation):
    """Resize an image using the specified interpolation."""
    return image.resize(size, interpolation)


def png_loader(path, resize=None, interpolation=Image.BICUBIC):
    filenames = sorted([os.path.join(path, f) for f in os.listdir(path)])
    images = [Image.open(f).convert('RGB') for f in filenames if f.endswith(".png")]

    if resize:
        images = [resize_image(img, resize, interpolation) for img in images]

    return [np.array(img) for img in images]


def nrrd_loader(path, resize=None, interpolation=Image.BICUBIC):
    data, header = nrrd.read(path)
    slices = [Image.fromarray(data[:, :, i]).convert('RGB') for i in range(data.shape[2])]

    if resize:
        slices = [resize_image(slice_, resize, interpolation) for slice_ in slices]

    return [np.array(slice_) for slice_ in slices]


def dicom_loader(path, resize=None, interpolation=Image.BICUBIC):
    filenames = sorted([os.path.join(path, f) for f in os.listdir(path)])
    dicom_files = [pydicom.dcmread(f, force=True) for f in filenames if f.endswith(".dcm")]
    slices = [Image.fromarray(f.pixel_array) for f in dicom_files if hasattr(f, "SliceLocation")]

    if resize:
        slices = [resize_image(slice_, resize, interpolation) for slice_ in slices]

    return [np.array(slice_) for slice_ in slices]


def raw_loader(filename, *, size_of_raw, channels, slices, resize=None, interpolation=Image.BICUBIC):
    with open(filename, "rb") as f:
        data_slices = []
        for _ in range(slices):
            raw_data = np.fromfile(f, 'uint8', size_of_raw[0] * size_of_raw[1] * channels)
            reshaped_raw_data = raw_data.reshape((size_of_raw[0], size_of_raw[1], channels))
            if channels == 1:
                reshaped_raw_data = raw_data.squeeze(-1)
            data_slices.append(Image.fromarray(reshaped_raw_data))

        if resize:
            data_slices = [resize_image(slice_, resize, interpolation) for slice_ in data_slices]

        return [np.array(slice_) for slice_ in data_slices]


