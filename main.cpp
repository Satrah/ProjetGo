#include <iostream>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>

#include "ImageLoader.h"
#include "Image.h"
#include "AlGo.h"
#include "PerspectiveFinder.h"


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
	int cornersOffset = 50;
	GOProject::ImageLoader loader;
	GOProject::PerspectiveFinder perspectiveFinder(cornersOffset);
	GOProject::AlGo go;
	int consecutiveSuccess = 0;
	// Calibrate
	while (consecutiveSuccess < 30)
	{
		char k = waitKey(10);
		if (k == 'q')
			return;
		if (k == 'p') // Pause
			continue;
		cap >> webcam;
		if (webcam.empty())
			continue;

		cvtColor(webcam, webcamGrey, CV_BGR2GRAY);

		perspectiveFinder.Load(webcamGrey);
		if (!perspectiveFinder.HomographyCalibrate())
		{
			consecutiveSuccess = 0;
			continue;
		}
		perspectiveFinder.HomographyTransform();
		loader.Load(perspectiveFinder);
		loader.DetectSquareForms(cornersOffset, webcamGrey.height() - cornersOffset, cornersOffset, webcamGrey.height() - cornersOffset);
		loader.DetectBoard2();

		loader.DetectIntersect();
		if (loader.FindHomographyWithDetectedRectangles())
		{
			/// READ FINISHED
			loader.ApplyHomography();
			++consecutiveSuccess;
		}
		else
			consecutiveSuccess = 0;
		imshow("HomographyTransformed", perspectiveFinder);
	}
	while (true)
	{
		if (k == 'q')
			return;
		if (k == 'p') // Pause
			continue;
		cap >> webcam;
		if (webcam.empty())
			continue;

		cvtColor(webcam, webcamGrey, CV_BGR2GRAY);
		perspectiveFinder.Load(webcamGrey);
		perspectiveFinder.HomographyTransform();
		loader.Load(perspectiveFinder);

		loader.DetectIntersect();
		loader.ApplyHomography();

		//loader.DebugDisplaySquares();

		//loader.DetectEllipse();

		createTrackbar("min line length", "Hough", &minLineLength, 100);
		createTrackbar("max gap", "Hough", &maxLineGap, 100);
		createTrackbar("max gaphoughTreshold", "Hough", &houghTreshold, 150);
		go.charge(loader);
		go.refresh(loader);
		go.affichePlateau();
		go.suggereCoup(loader);
	}
}