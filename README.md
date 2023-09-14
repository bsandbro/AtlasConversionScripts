Atlas Conversion Scripts (python3 package version)
========================

Introduction
-------------
This repository contains some scripts to help converting 3D volume data into a WebGL compatible 2D texture atlas.

The scripts allow you to generate an atlas from different types of volume data sources. This are the supported data types:
*	Common image formats like __PNG__, __JPEG__... which are supported by _PIL_
*	__DICOM__
*	__RAW__ 
*	__NRRD__ 

Also, there is a template script you can use to adapt it to your own volume data type.  

Documentation
--------------
You can found the necessary information about how you can use the scripts and how to visualize the atlas on the [wiki](https://github.com/VolumeRC/AtlasConversionScripts/wiki).

Set-up
------
This package utilizes a setup.py file to install the necessary dependencies and creates a cli command to run the scripts. 

To install the package, run the following command from the root directory of the repository:

```bash
pip install -e .
```

Running
-------
To run the script, you can use the following command after installation in your environment:

```bash
atlas-conversion <arguments>
```

Related Publication
-------------------
*	_John Congote, Alvaro Segura, Luis Kabongo, Aitor Moreno, Jorge Posada, and Oscar Ruiz. 2011_. __Interactive visualization of volumetric data with WebGL in real-time__. In Proceedings of the 16th International Conference on 3D Web Technology (Web3D '11). ACM, New York, NY, USA, 137-146.
