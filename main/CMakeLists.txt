cmake_minimum_required(VERSION 3.10)
project(example)

find_library(TENSORFLOW_LIB tensorflow HINT $ENV{HOME}/libtensorflow/lib)

set(CMAKE_CXX_STANDARD 17)
# find_package( OpenCV REQUIRED )
find_package( OpenCV REQUIRED PATHS /usr/local NO_DEFAULT_PATH)

add_executable(myUpscaler main.cpp ../src/Model.cpp ../src/Tensor.cpp ../src/myUtils.cpp)
target_include_directories(myUpscaler PRIVATE include $ENV{HOME}/libtensorflow/include)
target_link_libraries (myUpscaler "${TENSORFLOW_LIB}" ${OpenCV_LIBS} stdc++fs)
