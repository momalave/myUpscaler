// Created by Mario Malave

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Clamp values to a give lower and upper bound
void clip_by_value(Mat& mat, float lowerBound, float upperBound){ 
    vector<Mat> matc;
    split(mat, matc);
    min(max(matc[0], lowerBound), upperBound, matc[0]);
    min(max(matc[1], lowerBound), upperBound, matc[1]);
    min(max(matc[2], lowerBound), upperBound, matc[2]);
    merge(matc, mat);   
}

// Display and status bar
void drawStatus(int curFrame, int numFrames, float dur){
    int barWidth = 70;
    float progress = (float)curFrame/numFrames;
    int pos = barWidth * progress;
    cout << "[";    
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << int(progress * 100.0) << "% \t(" << curFrame << "/" << numFrames << ")    ";
    cout << dur << " ms\r";// << endl;    
    cout.flush();
}

// Add audio to upsampled video
void processAudio(string inputDir, string outputDir){ 
    size_t extPos = outputDir.find_last_of(".");
    string tempDir = outputDir.substr(0,extPos) + "_temp.mp4";
    tempDir = tempDir.substr(extPos);
    
    //add audio using ffmpeg library
    system(("ffmpeg -y -i " + inputDir + " -i " + outputDir + 
            " -c copy -map 0:a:0 -map 1:v:0 -shortest -c copy -flags global_header -loglevel fatal " + tempDir).c_str());
        
    // replace output upsampled video with output of ffmpeg
    system(("rm " + outputDir).c_str());
    system(("mv " + tempDir + " " + outputDir).c_str());
}