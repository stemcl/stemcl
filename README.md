# stemcl
A GPU-accelerated implementation of the STEM simulation code by E. Kirkland optimized for high resolution simulation of large specimen. Stemcl uses the OpenCL API to leverage the full power of heterogenous systems including GPUs and CPUs from all major vendors.

## Running stemcl on Amazon EC2
We provide an EC2 AMI with pre-installed versions of stemcl and all necessary drivers. This image can be used with the p2 and p3 GPU compute instance types. [Launch instance](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-16993a6c).

## Installation

### Requirements
To build stemcl, you will need at least version 3.5 of cmake, a recent version of clfft, and the OpenCL headers and library. You may see some deprecation warnings because stemcl was programmed against a pretty old version of the OpenCL standard. This is normal.

### Ubuntu 16.04 or later
```
sudo apt update
sudo apt install build-essential cmake libclfft-dev libtiff-dev
git clone https://github.com/stemcl/stemcl.git && cd stemcl
mkdir build && cd build
cmake ..
make
sudo make install
```

### MacOS 10.12 or later
Please install [homebrew](https://brew.sh) and Xcode (or the Xcode Command Line Tools) first.

```
brew install clfft cmake
git clone https://github.com/stemcl/stemcl.git
cd stemcl
mkdir build && cd build
cmake ..
make
sudo make install
```

## Usage
To run stemcl you will need at least one available OpenCL platform that supports your hardware. This generally boils down to installing the correct driver for your CPU or GPU. OpenCL drivers are part of the standard installation on macOS. For Ubuntu, the download locations of the most popular OpenCL implementations are listed below.

| CPU | GPU | Vendor | Download/Installation |
|-----|-----|--------|-----------------------|
| Yes | Yes | AMD    | |
|     | Yes | Nvidia | `apt-get install nvidia-opencl-icd-352` |
|     | Yes | Intel  | `apt-get install beignet-opencl-icd`
| Yes | Yes | Intel  | https://software.intel.com/en-us/articles/opencl-drivers

To start a simulation with stemcl, copy your specimen definition (in `.xyz` format) and the `parameter.dat` file to a working directory of your choice. There is a sample simulation definition in the `sample/` directory to get you started. Then start stemcl with `stemcl <opencl platform> <device id> <simulation directory>`. To use multiple OpenCL devices, simply start multiple instances of stemcl. For example, if you have two Nvidia GPUs:

```
stemcl nvidia 0 simulation &
stemcl nvidia 1 simulation &
```

If you simulate very large specimen you may need to alter the settings in the last line of the `parameter.dat` file:

- If you run out of GPU memory, decrease `num_parallel`. This is mostly needed for big transmission functions. For transmission sizes greater than 4096x4096 this value can be reduced to 2 without a performance penalty compared to the default value of 16.
- If you run out of host memory, set `use_hdd` to 1. This will save transmission functions to disk instead of keeping them in memory. Running out of host memory can happen if you use many slices or very high resolutions for the transmission or probe functions. This will slow down you simulation! Consider buying more RAM instead.

## Benchmarks
All times are hh:mm:ss.

Count | GPU Type | 512x512  |1024x1024 |2048x2048 |4096x4096 |8192x8192  |
-----:+----------+---------:+---------:+---------:+---------:+----------:+
1     | K80      | 00:01:28 |          |          |          |           |
2     | K80      | 00:00:46 |          |          |          |           |
4     | K80      | 00:00:24 |          |          |          |           |
8     | K80      | 00:00:15 | 00:01:00 | 00:04:57 | 00:27:20 | 02:21:17  |
16    | K80      | 00:00:14 | 00:00:39 | 00:02:51 | 00:14:33 | 13:02:42  |
1     | Titan X  | 00:00:46 | 00:03:00 | 00:13:13 |          |           |
1     |Titan X (Pascal)| 00:00:37 | 00:02:10 | 00:10:27 |          |           |
1     | V100     | 00:00:29 | 00:01:41 | 00:10:09 |          |           |
4     | V100     | 00:00:09 | 00:00:31 | 00:03:01 | 00:21:57 |           |

## Seeing results
stemcl results are written in a binary format that can be converted to TIFF images with `stemcl2tiff`, which has been moved to a separate repository at [stemcl/stemcl2tiff](https://github.com/stemcl/stemcl2tiff).

## License
Copyright 2017 Manuel Radek, Jan-Gerd Tenberge


This program is free software: you can redistribute it and/or modify 
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The following publication is the canoncial reference to use for citing stemcl. A citation is mandatory if you want to publish data produced with stemcl. 
Please also give the URL of the stemcl repository in your paper, namely https://github.com/stemcl/stemcl.


NO WARRANTY 

THIS PROGRAM IS PROVIDED AS-IS WITH ABSOLUTELY NO WARRANTY
OR GUARANTEE OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
INCLUDING BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
IN NO EVENT SHALL THE AUTHOR BE LIABLE
FOR DAMAGES RESULTING FROM THE USE OR INABILITY TO USE THIS
PROGRAM (INCLUDING BUT NOT LIMITED TO LOSS OF DATA OR DATA
BEING RENDERED INACCURATE OR LOSSES SUSTAINED BY YOU OR
THIRD PARTIES OR A FAILURE OF THE PROGRAM TO OPERATE WITH
ANY OTHER PROGRAM). 



