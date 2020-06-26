# Upscaling Model Implementation

C++ implementation of a 4x upscaler

Command line options:

```sh
--input-file: Path to the input video file that will get processed
--output-file: Path to the output video file that will get created
```

## Installation Instructions

Since it uses TensorFlow C API you just have to [download it](https://www.tensorflow.org/install/lang_c).  

You can either install the library system wide by following the tutorial on the Tensorflow page or you can place the contents of the archive
in a folder called `libtensorflow` in the home directory.

Afterwards, you can run the examples:

```sh
git clone git@github.com:serizba/cppflow.git
cd cppflow/examples/load_model
mkdir build
cd build
cmake ..
make
./example
```

## CppFlow

https://github.com/serizba/cppflow
https://github.com/dhiegomaga/cppflow

Run TensorFlow models in c++ without Bazel, without TensorFlow installation and without compiling Tensorflow.

CppFlow uses Tensorflow C API to run the models, meaning you can use it without installing Tensorflow and without compiling the whole TensorFlow repository with bazel, you just need to download the C API.
