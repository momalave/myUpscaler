// Created by Mario Malave

#ifndef MYUPSCALER_MYUTILS_H
#define MYUPSCALER_MYUTILS_H

using namespace cv;
using namespace std;

// Clamp values to a give lower and upper bound
void clipByValue(Mat& mat, float lowerBound, float upperBound);

// Display and status bar
void drawStatus(int curFrame, int numFrames, float dur);

// Add audio to upsampled video
void processAudio(string inputDir, string outputDir);

#endif //MYUPSCALER_MYUTILS_H
