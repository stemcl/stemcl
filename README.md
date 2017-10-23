# stemcl

## Installation
To build stemcl on Ubuntu 16.04 or later:

```
sudo apt-get install build-essential cmake libclfft-dev libtiff-dev
git clone https://github.com/stemcl/stemcl.git
cd stemcl
mkdir build && cd build
cmake ..
make 
```

You may see some deprecation warnings because stemcl was programmed against a pretty old version of the OpenCL standard. This is normal. To run stemcl you will need at least one available OpenCL platform that supports your hardware. This generally boils down to installing the correct driver for your CPU or GPU. OpenCL drivers are part of the standard installtion on macOS. For Ubuntu, the download locations of the most popular OpenCL implementations are listed below.

| CPU | GPU | Vendor | Download/Installation |
|-----|-----|--------|-----------------------|
| Yes | Yes | AMD    | |
|     | Yes | Nvidia | `apt-get install nvidia-opencl-icd-352` |
|     | Yes | Intel  | `apt-get install beignet-opencl-icd`
| Yes | Yes | Intel  | https://software.intel.com/en-us/articles/opencl-drivers

## Usage
To start a simulation with stemcl, copy your specimen definition (in `.xyz` format) and the `parameter.dat` file to a working directory of your choice. There is a sample simulation definition in the `assets` directory to get you started. Then start stemcl with `stemcl <opencl platform> <device id> <simulation directory>`. To use multiple OpenCL devices, simply start multiple instances of stemcl. For example, if you have two Nvidia GPUs:

```
stemcl nvidia 0 simulation &
stemcl nvidia 1 simulation &
```

You can get a list of all availble devices by running stemcl with no arguments.

## Seeing results
stemcl results are written in a binary format that can be converted to TIFF images with `stemcl2tiff`, which is part of this repository. To generate a `.tiff` output of an 1024x1024 stemcl output with `.stemcl` file ending and two detectors:
```
stemcl2tiff 1024 1024 .stemcl 2
```
