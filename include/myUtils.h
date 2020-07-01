// Created by Mario Malave

#ifndef MYSCALER_MYUTILS_H
#define MYSCALER_MYUTILS_H

using namespace cv;
using namespace std;

// Clamp values to a give lower and upper bound
void clipByValue(Mat& mat, float lowerBound, float upperBound);

// Display and status bar
void drawStatus(int curFrame, int numFrames, float dur);

// Add audio to upsampled video
void processAudio(string inputDir, string outputDir);

#endif //MYSCALER_MYUTILS_H
