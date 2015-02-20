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
	int minLineLength = 50;
	int maxLineGap = 20;
	int houghTreshold = 50;
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
		loader.DetectLinesHough(houghTreshold, minLineLength, maxLineGap);
		loader.BuildHoughLinesHistogram();
		loader.FilterHoughLines();
		Image<uchar> blah = loader.DisplayHoughLines("Hough");
		loader.DisplayHoughLinesOrientation();
		loader.DisplayTransformedImage();
		createTrackbar("min line length", "Hough", &minLineLength, 100);
		createTrackbar("max gap", "Hough", &maxLineGap, 100);
		createTrackbar("max gaphoughTreshold", "Hough", &houghTreshold, 150);
	}

	waitKey();
}