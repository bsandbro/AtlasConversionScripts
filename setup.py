from setuptools import setup

setup(
    name='atlas_conversion',
    version='0.1.0',
    packages=['atlas_conversion'],
    install_requires=['numpy',
                      'scipy',
                      'h5py',
                      'dask',
                      'pillow'
                      'pydicom'],
    entry_points={
        'console_scripts': [
            'atlas_conversion = atlas_conversion.__main__:main'
        ]
    })
