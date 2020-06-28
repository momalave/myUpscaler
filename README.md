# Upscaling Model

C++ implementation of a 4x upscaler

Command line options:

```sh
--input-file: Path to the input video file that will get processed
--output-file: Path to the output video file that will get created
--audio-flag: set to 1 to include audio, default is 0
```

## Installation Instructions

Download TensorFlow C API [here](https://www.tensorflow.org/install/lang_c).  

Install the library system wide by following the tutorial on the Tensorflow page or place the contents of the archive
in the home directory in a folder called `libtensorflow`. The `CMakeLists.txt` file points to the `libtensorflow` folder in the home directory.

Then, git clone the repository and compile the code using cmake:

```sh
git clone https://github.com/momalave/myScaler.git
cd myScaler/main
mkdir build
cd build
cmake ..
make
./myScaler --input-file /path/to/video/input --output-file /path/to/video/out --audio-flag <optional flag to include audio>
```

## CppFlow

In this project the cppFlow library was used to load pre-trained models in C/C++. This was made possible by the work of [serzba](https://github.com/serizba/cppflow) and [dhiegomaga](https://github.com/dhiegomaga/cppflow).

Run TensorFlow models in c++ without Bazel, without TensorFlow installation and without compiling Tensorflow.

CppFlow uses Tensorflow C API to run the models, meaning you can use it without installing Tensorflow and without compiling the whole TensorFlow repository with bazel, you just need to download the C API.
