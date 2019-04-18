# RTOW-OptiX (producing visualizations)
## Overview

This project is based off of Ingo Wald's Optix Version of the 'final chapter'
example in Pete Shirley's "Ray Tracing in one Week-End" series. The code was expanded to 
handle different tensor data files in csv format. Additional transforms were added to
handle creating ellipsoids and superquadratics based off of the tensor data.

## Setup
To build this project, you need

- a install of CUDA, preferably CUDA 10. Make sure to put your CUDA
  binary directory into your path.

- a install of OptiX, preferably (ie, tested with) OptiX 5.1.1. Under
  Linux, put a ```export OptiX_INSTALL_DIR=...``` into your
  ```.bashrc```.

- normal compiler and build tools - gcc, clang, etc.

- cmake, version 2.8 at least.

## Building

This project is built with cmake. On linux, simply create a build
directory, and start the build with with ccmake:

   mkdir build
   cd build
   cmake ..
   make

Assuming you have nvcc (CUDA) in your path, and have set a
```OptiX_INSTALL_DIR``` environment variable to point to the OptiX
install dir, everything should be configured automatically.

On windows, you'll have to use the cmake gui, and make sure to set the
right paths for optix include dirs, optix paths, etc.


## Running

Run ./finalChapter_iterative binary. The format to run it is 
```
./finalChapter_iterative <configuration file>.yaml whitted|pathtracing
```
Use the keyword "whitted" for Whitted-style ray-tracing, and "pathtracing"
for Path-traced ray-tracing images.

