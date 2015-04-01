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
	GOProject::ImageLoader loader;
	GOProject::PerspectiveFinder perspectiveFinder;
	GOProject::AlGo go;
	while (true)
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
		perspectiveFinder.HomographyTransform();
		loader.Load(perspectiveFinder);
		loader.DetectSquareForms();
		loader.DetectBoard2();

		// Lets find an homography in the found rectange :)
		/*
		loader.DetectLinesHough(houghTreshold, minLineLength, maxLineGap);
		loader.BuildHoughLinesHistogram();
		//loader.DisplayHoughLinesOrientation();
		//loader.DisplayHoughLines("Hough");
		loader.FilterVerticalLines();
		loader.FindBestHomography();
		loader.DisplayTransformedImage();
		*/
		loader.DetectIntersect();
		if (loader.FindHomographyWithDetectedRectangles())
		{
			/// READ FINISHED
			loader.ApplyHomography();

			imshow("Transformed Image", loader.GetImage());
		}

		loader.DebugDisplaySquares();

		loader.DetectEllipse();

		createTrackbar("min line length", "Hough", &minLineLength, 100);
		createTrackbar("max gap", "Hough", &maxLineGap, 100);
		createTrackbar("max gaphoughTreshold", "Hough", &houghTreshold, 150);
		go.charge(loader);
		go.affichePlateau();
		go.suggereCoup(loader);
	}

	waitKey();
}