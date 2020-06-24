#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <thread>
#include <cmath>

#include "../include/Model.h"
#include "../include/Tensor.h"

using namespace cv;
using namespace std;

static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> argument(s)\n"
              << "Options:\n"
              << "\t-h,--help      \tUsage information\n"
              << "\t--input-file   \tPath to the input video file that will get processed\n"
              << "\t--output-file  \tPath to the output video file that will get created\n"
              << std::endl;
}



int main(int argc, char* argv[]){


    if (argc < 3) {
        show_usage(argv[0]);
        return 1;
    }
    vector <string> sources;
    string inputDir, outputDir;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } else if (arg == "--input-file") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                inputDir = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  cerr << "--input-file option requires one argument." << endl;
                return 1;
            }
        } else if (arg == "--output-file") {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                outputDir = argv[++i]; // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the destination option.
                  cerr << "--output-file: option requires one argument." << endl;
                return 1;
            }
        } else {
            sources.push_back(argv[i]);
        }
    }


  // Decalre variables
  int scalefactor = 4;   // based on the trained model
  Mat image, flat, processed_image;
  vector<float> predictions;

  int numFrames = 0, barWidth = 70, pos = 0, curFrame = 0;
  float progress = 0.0, dur = 0.0, sumTime = 0.0, sumSqTime = 0.0;
  Mat outframe;

  // Initialize Model
  Model m("../upscaler_model");
  // Input and output Tensors
  Tensor input(m, "serving_default_input_5");
  Tensor prediction(m, "StatefulPartitionedCall");

  // Create a VideoCapture object and open the input file
  VideoCapture cap(inputDir);
  // Check if file opened successfully
  if(!cap.isOpened()){
    cout << "Error opening file" << endl;
    return -1;
  }
  // Get number of total frames
  numFrames = int(cap.get(CAP_PROP_FRAME_COUNT));

  // Create a VideoWriter object
  double fps = cap.get(CAP_PROP_FPS);
  int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
  //numFrames = fps;

  // Create a VideoWriter object for 4x upSampler output.mp4
  VideoWriter video;
  //Use the AV_CODEC_ID_H264 , 0x21 to save using H.264
  //Source: https://stackoverflow.com/questions/34024041/writing-x264-from-opencv-3-with-ffmpeg-on-linux
  video.open(outputDir, 0x21, fps, Size(scalefactor*frame_width,scalefactor*frame_height), true);

  cout << "----- Processing Video Frames  -----" << endl ;

  // Capture the first frame
  cap >> image;

  // Obtain image dimensions
  int rows = image.rows;
  int cols = image.cols;
  int channels = image.channels();
  int total = image.total();

  // Initialize input data
  vector<float> img_data(rows*cols*channels);

  // Iterate over each frame
  while(1){
    //Get current frame number
    curFrame = int(cap.get(CAP_PROP_POS_FRAMES));

    // Convert to CV_32F, 32 bit floating point number
    image.convertTo(image, CV_32F);

    // Flatten the images and store it in a vector
    // Source: https://stackoverflow.com/a/56600115/2076973
    flat = image.reshape(1, image.total() * channels);
    img_data = image.isContinuous()? flat : flat.clone();

    // Feed data to input tensor
    input.set_data(img_data, {1, rows, cols, channels});

    // Display and status bar
    progress = (float)curFrame/numFrames;
    cout << "[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
      if (i < pos) cout << "=";
      else if (i == pos) cout << ">";
      else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "% \t(" << curFrame << "/" << numFrames << ")    ";

    // Run inference on the model
    auto start = chrono::steady_clock::now();
    m.run(input, prediction);
    auto end = chrono::steady_clock::now();

    // Used for calculating the mean and std of the inference time
    dur = chrono::duration_cast<chrono::milliseconds>(end - start).count();
    sumTime += dur;
    sumSqTime += pow(dur,2.0);

    cout << dur << " ms\r" << endl;
    //cout.flush();

    // Get tensor with predictions
    predictions = prediction.Tensor::get_data<float>();

    // unflatten
    processed_image = Mat(scalefactor*rows, scalefactor*cols, image.type(), predictions.data());

    // Convert to CV_32F, unsigned 8bit/pixel - ie a pixel can have values 0-255
    processed_image.convertTo(processed_image, CV_8U);
    // Write the frame into the file 'outcpp.avi'
    video.write(processed_image);

    // Obtain next frame
    cap >> image;
    // Exit if the frame is empty
    if (image.empty())
    break;

    if (curFrame == numFrames) {break;}
  }

  //cout << numframes << endl;
  // When everything done, release the video capture object
  cap.release();
  video.release();

  // Closes all the frames
  destroyAllWindows();

  cout << "\n";
  cout << "----- Inference Time Statistics ----- " << endl ;
  cout << "Total Time: " << sumTime << " ms" << endl ;
  //Calculate the average
  cout << "Average: " << sumTime/numFrames << " ms" << endl ;
  //Calculate the standard deviation
  //Source: https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
  cout << "Standard Deviation: " << sqrt((sumSqTime -(pow(sumTime,2.0)/numFrames))/numFrames) << " ms" << endl ;

  return 0;
}


