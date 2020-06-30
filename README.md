# Upscaling Model

C++ implementation of a 4x upscaler

Command line options:

```sh
--input-file: Path to the input video file that will get processed
--output-file: Path to the output video file that will get created
--audio-flag: Set to 1 to include audio, default is 0
```

## Installation Instructions (Linux)

### Dependencies:

Download TensorFlow C API [here](https://www.tensorflow.org/install/lang_c). Install the library system wide by following the tutorial on the Tensorflow page or place the contents of the archive in the home directory in a folder called `libtensorflow`. The `CMakeLists.txt` file points to the `libtensorflow` folder in the home directory.

Additionally, [Tensorflow 2.2](https://www.tensorflow.org/install/pip), [OpenCV 4.4](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html), and [FFmpeg](https://ffmpeg.org/download.html) for video/audio processing.

Next, git clone the repository:
```sh
git clone https://github.com/momalave/myScaler.git
```

Then, compile the code using cmake:
```
cd myScaler/main
mkdir build
cd build
cmake ..
make
```

Run the code using the appropiate arguments: 
```
./myScaler --input-file /path/to/video/input --output-file /path/to/video/out --audio-flag <optional flag to include audio>
```

## CppFlow Library

In this project, the [CppFlow library](https://github.com/serizba/cppflow) was used to load pre-trained models in C/C++. CppFlow enables TensorFlow models to run in C/C++ without Bazel, without a TensorFlow installation, and without compiling Tensorflow.
