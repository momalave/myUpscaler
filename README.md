# Upscaling Model

C++ implementation of a deep learning model that upscales RGB images using a factor of 4x (i.e. 270p to 1080p). The model was pre-trained and saved using the [SavedModel](https://www.tensorflow.org/guide/saved_model) format in [Tensorflow](https://www.tensorflow.org/install/pip).

Command line options:

```sh
--input-file: Path to the input video file that will get processed
--output-file: Path to the output video file that will get created
--model-path: Path to the pre-trained model, default path is "../upscaler_model"
--audio-flag: Flag to process audio (audio stream in input video required)
```

## Installation Instructions (Linux)

### Dependencies:

Download TensorFlow C API [here](https://www.tensorflow.org/install/lang_c). Install the library system wide by following the tutorial on the Tensorflow page or place the contents of the archive in the home directory in a folder called `libtensorflow`. The `CMakeLists.txt` file points to the `libtensorflow` folder in the home directory. Additionally, install [Tensorflow 2.2](https://www.tensorflow.org/install/pip), [OpenCV 4.4](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html), and [FFmpeg](https://ffmpeg.org/download.html) for video/audio processing.

Currently, the repository is private. If access is needed, please send a request to momalave@gmail.com.

Next, git clone the repository:
```sh
git clone https://github.com/momalave/myScaler.git
```

Then, compile the code using cmake:
```sh
cd myScaler/main
mkdir build
cd build
cmake ..
make
```

### Pre-trained model included in:
```sh
myScaler/main/upscaler_model
```

### Run using the appropiate arguments:
```sh
./myScaler --input-file /path/to/video/input --output-file /path/to/video/out --model-path <optional, path/to/model/folder, default "../upscaler_model"> --audio-flag <optional flag, default audio processing off>
```

#### Example 1 
(process big_buck_bunny.mp4, save as upsampled_big_buck_bunny.mp4, and use the default model path model "../upscaler_model"): 
```sh
./myScaler --input-file big_buck_bunny.mp4 --output-file upsampled_big_buck_bunny.mp4
```
#### Example 2 
(process big_buck_bunny.mp4 and save as upsampled_big_buck_bunny.mp4 using the model in the "different_model" folder): 
```sh
./myScaler --input-file big_buck_bunny.mp4 --output-file upsampled_big_buck_bunny.mp4 --model-path different_model
```
#### Example 3 
(process toy_story.mp4, save as upsampled_toy_story.mp4, use the default model path model "../upscaler_model)", and include the audio: 
```sh
./myScaler --input-file toy_story.mp4 --output-file upsampled_toy_story.mp4 --audio-flag
```

## CppFlow Library

In this project, the [CppFlow library](https://github.com/serizba/cppflow) was used to load pre-trained models in C/C++. CppFlow enables TensorFlow models to run in C/C++ without Bazel, without a TensorFlow installation, and without compiling Tensorflow.
