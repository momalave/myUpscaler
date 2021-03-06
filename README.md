# Upscaling Model

C++ implementation of a deep learning model that upscales RGB images using a factor of 4x (i.e. 270p to 1080p). The model was pre-trained and saved using the [SavedModel](https://www.tensorflow.org/guide/saved_model) format in [Tensorflow](https://www.tensorflow.org/install/pip).

Command line options:

```sh
--input-file: Path to the input video file that will get processed
--output-file: Path to the output video file that will get created
--model-path: Path to the pre-trained model, default path is "../upscaler_model"
--audio-flag: Flag to process audio (audio stream in input video required), default audio processing is off
```

## Installation and Compiling Instructions (Linux)

### Dependencies:
To load pre-trained models in C++, the [CppFlow library](https://github.com/serizba/cppflow) was used. CppFlow enables TensorFlow models to run in C++ without Bazel, without a TensorFlow installation, and without compiling TensorFlow.

CppFlow requires the TensorFlow C API. Download the appropriate (i.e., Linux CPU only or Linux CPU/GPU support) TensorFlow C API [here](https://www.tensorflow.org/install/lang_c). Then, put the contents of the archive in the home directory in a folder called `libtensorflow` (or install the library system wide by following the tutorial on the TensorFlow page). The `myUpscaler/main/CMakeLists.txt` file searches for the `libtensorflow` folder in the home directory. Additionally, install the appropriate [TensorFlow 2.2](https://www.tensorflow.org/install/pip), [CUDA 11.0](https://developer.nvidia.com/cuda-downloads), [cuDNN 7.5](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html), [OpenCV 4.4](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html), and [FFmpeg](https://ffmpeg.org/download.html) for video/audio processing.

For development, an Amazon Web Services (AWS) instance was used with a [Deep Learning Amazon Machine Image (DLAMI) (Ubuntu 16.04/18.04) Version 30.0](
https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html). This comes preconfigured with many of the dependencies (i.e., TensorFlow, OpenCV, FFmpeg, CUDA, and cuDNN) and makes setting up the environment substantially easier.

Next, git clone the repository:
```sh
git clone https://github.com/momalave/myUpscaler.git
```

Then, compile the code using cmake:
```sh
cd myUpscaler/main
mkdir build
cd build
cmake ..
make
```

## Execution Instructions (Linux)

### Pre-trained model included in:
```sh
myUpscaler/main/upscaler_model
```

### Run using the appropriate arguments:
```sh
./myUpscaler --input-file /path/to/video/input --output-file /path/to/video/out --model-path <optional, path/to/model/folder, default "../upscaler_model"> --audio-flag <optional flag, default audio processing is off>
```

#### Example video files:
[big_buck_bunny.mp4](https://www.dropbox.com/s/9cfcxflggejz6k0/big_buck_bunny.mp4?dl=0)

[toy_story.mp4](https://www.dropbox.com/s/gtphbavjvo0813d/toystory.mp4?dl=0)

#### Example 1: 
Process big_buck_bunny.mp4, save as upsampled_big_buck_bunny.mp4, and use the default model path "../upscaler_model": 
```sh
./myUpscaler --input-file big_buck_bunny.mp4 --output-file upsampled_big_buck_bunny.mp4
```
#### Example 2:
Process big_buck_bunny.mp4, save as upsampled_big_buck_bunny.mp4, and use the model in the "different_model" folder: 
```sh
./myUpscaler --input-file big_buck_bunny.mp4 --output-file upsampled_big_buck_bunny.mp4 --model-path different_model
```
#### Example 3:
Process toy_story.mp4, save as upsampled_toy_story.mp4, use the default model path model "../upscaler_model", and include the audio: 
```sh
./myUpscaler --input-file toy_story.mp4 --output-file upsampled_toy_story.mp4 --audio-flag
```
