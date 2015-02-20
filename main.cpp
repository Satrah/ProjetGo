#include <iostream>

#include <opencv2/highgui/highgui.hpp>

#include "ImageLoader.h"
#include "Image.h"

using namespace GOProject;
using namespace std;
using namespace cv;

void TestHoughLinesFromWebcam();

int main()
{
	TestHoughLinesFromWebcam();
    return 0;
}

void TestHoughLinesFromWebcam()
{
	VideoCapture cap;
	cap.open(0);
	Mat webcam;
	Image<uchar> webcamGrey;
	Mat harrisCorners;
	while (waitKey(1) != 'q')
	{
		cap >> webcam;
		if (webcam.empty())
			continue;

		cvtColor(webcam, webcamGrey, CV_BGR2GRAY);

		GOProject::ImageLoader loader;
		loader.Load(webcamGrey);
		loader.DetectLinesHough();
		loader.DisplayHoughLines("Hough");
	}

	waitKey();
}