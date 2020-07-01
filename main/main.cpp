// Created by Mario Malave

#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>
#include <cmath>

#include "../include/Model.h"
#include "../include/Tensor.h"
#include "../include/myUtils.h"

using namespace cv;
using namespace std;


// Argument information
static void show_usage(string name){   
    std::cerr << "Usage: " << name << " <option(s)> argument(s)\n"
              << "Options:\n"
              << "\t-h,--help      \tUsage information\n"
              << "\t--input-file   \tPath to the input video file that will get processed\n"
              << "\t--output-file  \tPath to the output video file that will get created\n"
              << "\t--model-path   \tPath to the pre-trained model, default path is \"../upscaler_model\"\n"
              << "\t--audio-flag   \tInclude flag to process audio (audio stream in input video required)\n"
              << std::endl;
}

static int checkArgs(string args){
    string options = "--input-file--output-file--model-path--audio-flag-h--help";
    return (options.find(args) != std::string::npos) ? 1 : 0;
}

int main(int argc, char* argv[]){
    //Argument handling
    if (argc < 3) {
        show_usage(argv[0]);
        return 1;
    }
    
    string inputDir, outputDir, arg;
    string modelPath = "../upscaler_model";
    int audioFlag = 0;
    
    for (int i = 1; i < argc; ++i) {
        arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } else if (arg == "--input-file") {
            if (i + 1 < argc) { // check if at the end of argv
                inputDir = argv[++i]; // increment i to get the next option in argv[i]
            } else {
                cerr << "--input-file: option requires one argument." << endl;
                return 1;
            }
 
            if (checkArgs(inputDir)){ // no argument in the the option
                  cerr << "--input-file option requires one argument." << endl;
                return 1;
            }
        } else if (arg == "--output-file") {
            if (i + 1 < argc) { // check if at the end of argv
                outputDir = argv[++i]; // increment i to get the next option in argv[i]
            } else {
                cerr << "--output-file: option requires one argument." << endl;
                return 1;
            }
            
            if (checkArgs(outputDir)) { // no argument in the the option
                  cerr << "--output-file: option requires one argument." << endl;
                return 1;
            }
        } else if (arg == "--model-path") {
            if (i + 1 < argc) { // check if at the end of argv
                modelPath = argv[++i]; // increment i to get the next option in argv[i]
            } else {
                cerr << "--model-path: option requires one argument." << endl;
                return 1;
            }
            
            if (checkArgs(modelPath)){ // no argument in the the option
                  cerr << "--model-path: option requires one argument." << endl;
                return 1;
            }
        } else if (arg == "--audio-flag") {
            if ((i + 1 < argc) && !checkArgs(argv[i+1])){ // an argument was placed after --audio-flag, at the end 
                cerr << arg << ": does not take arguments, please follow the usage format:" << endl;
                show_usage(argv[0]);
                return 1;
            } 
            audioFlag = 1;  
        } else {
            cerr << argv[i] << " is not a valid option/argument, please follow the usage format:" << endl;
            show_usage(argv[0]);
            return 1;
        }
    }
    
    // Decalre variables  
    int scalefactor = 4;   // based on the trained 4x upSampler model
    Mat image, flat, processed_image;
    vector<float> predictions;
    int numFrames = 0, curFrame = 0;
    float dur = 0.0, sumTime = 0.0, sumSqTime = 0.0;

    // Initialize Model  
    Model m(modelPath);
    // Input and output Tensors
    Tensor input(m, "serving_default_input_5");
    Tensor prediction(m, "StatefulPartitionedCall");

    // Create a VideoCapture object and open the input file  
    VideoCapture cap(inputDir);
    // Check if file opened successfully
    if(!cap.isOpened()){
        cout << "Error opening input file, please check path..." << endl;
        return -1;
    }
  
    // Get frame paraments
    numFrames = int(cap.get(CAP_PROP_FRAME_COUNT));
    double fps = cap.get(CAP_PROP_FPS);  
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    //numFrames = fps; //used for processing 1 second

    // Create a VideoWriter object for 4x upSampler output
    VideoWriter video;
    //Use the AV_CODEC_ID_H264 , 0x21 to save using H.264
    //Source: https://stackoverflow.com/questions/34024041/writing-x264-from-opencv-3-with-ffmpeg-on-linux
    video.open(outputDir, 0x21, fps, Size(scalefactor*frame_width,scalefactor*frame_height), true);
    if (!video.isOpened())
    {
        cout << "Error creating ouput file, please check path..." << endl;
        return -1;
    }

    cout << "Processing Video Frames..." << endl ;  
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
    while(!image.empty()){
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
        
        // Run inference on the model
        auto start = chrono::steady_clock::now();
        m.run(input, prediction);
        auto end = chrono::steady_clock::now();

        // Used for calculating the mean and std of the inference time    
        dur = chrono::duration_cast<chrono::milliseconds>(end - start).count();
        sumTime += dur;
        sumSqTime += pow(dur,2.0);

        // Display status bar 
        drawStatus(curFrame, numFrames, dur);

        // Get tensor with predictions    
        predictions = prediction.Tensor::get_data<float>();
        // Unflatten processed_image
        processed_image = Mat(scalefactor*rows, scalefactor*cols, image.type(), predictions.data());    
    
        // Postprocess output of the model
        // Clip from 0 to 255, y = tf.clip_by_value(y, 0, 255), aka clamp operation
        clipByValue(processed_image, 0, 255);
        // convertTo will Round, y = tf.round(y),  
        // and convert to CV_8U, unsigned 8bit/pixel - ie a pixel can have values 0-255, y = tf.cast(y, tf.uint8)   
        processed_image.convertTo(processed_image, CV_8U);   
        //cout << processed_image << endl;
        
        // Write the frame into the output video file
        video.write(processed_image);
        
        // Obtain next frame
        cap >> image;
    
        // Exit if current frame is the last
        if (curFrame == numFrames) {
            cout << "\n";
            break;
        }
    }
    
    // Release the videocapture and videowriter objects
    cap.release();
    video.release();
    
    // if audioFlag == 1, add audio to upscaled video
    if(audioFlag){
        cout << "\nProcessing audio..." << endl;
        processAudio(inputDir, outputDir);
    }
    
    // Display model inference statistics
    cout << "\n----- Inference Time Statistics ----- " << endl ;
    cout << "Total Time: " << sumTime << " ms" << endl ;
    //Calculate the average  
    cout << "Average: " << sumTime/numFrames << " ms" << endl ;
    //Calculate the standard deviation
    //Source: https://math.stackexchange.com/questions/20593/calculate-variance-from-a-stream-of-sample-values
    cout << "Standard Deviation: " << sqrt((sumSqTime -(pow(sumTime,2.0)/numFrames))/numFrames) << " ms" << endl ;
    return 0;
}
