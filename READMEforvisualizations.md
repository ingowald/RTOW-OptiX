# RTOW-OptiX (producing visualizations)
## Overview

This project is based off of Ingo Wald's Optix Version of the 'final chapter'
example in Pete Shirley's "Ray Tracing in one Week-End" series. The code was expanded to 
handle different tensor data files in csv format. Additional transforms were added to
handle creating ellipsoids and superquadratics based off of the tensor data. Follow the
instructions in the README to setup everything. Then follow the instructions here to
produce visualizations

## Running

Run ./finalChapter_iterative binary. The format to run it is 
```
./finalChapter_iterative <configuration file>.yaml whitted|pathtracing
```
Use the keyword "whitted" for Whitted-style ray-tracing, and "pathtracing"
for Path-traced ray-tracing images. The configuration file should have the
file path to the desired dataset in it.

