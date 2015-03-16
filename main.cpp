#include <iostream>
#include <ctime>

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
	GOProject::ImageLoader loader;
	while (waitKey(10) != 'q')
	{
		cap >> webcam;
		if (webcam.empty())
			continue;

		cvtColor(webcam, webcamGrey, CV_BGR2GRAY);

		loader.Load(webcamGrey);
		loader.DetectLinesHough(houghTreshold, minLineLength, maxLineGap);
		loader.BuildHoughLinesHistogram();
		//loader.DisplayHoughLinesOrientation();
		//loader.DisplayHoughLines("Hough");
		loader.FilterVerticalLines();
		loader.FindBestHomography();
		//loader.DisplayTransformedImage();
		loader.ClearBadLines();
		loader.DisplayVerticalAndHorizontalLines("HoughCleared");
		createTrackbar("min line length", "Hough", &minLineLength, 100);
		createTrackbar("max gap", "Hough", &maxLineGap, 100);
		createTrackbar("max gaphoughTreshold", "Hough", &houghTreshold, 150);



		Mat threshold_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		/// Detect edges using Threshold
		threshold(webcamGrey, threshold_output, 100, 255, THRESH_BINARY);
		/// Find contours
		findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		/// Find the rotated rectangles and ellipses for each contour
		vector<RotatedRect> minRect(contours.size());
		vector<RotatedRect> minEllipse(contours.size());

		for (int i = 0; i < contours.size(); i++)
		{
			minRect[i] = minAreaRect(Mat(contours[i]));
			if (contours[i].size() > 5)
			{
				minEllipse[i] = fitEllipse(Mat(contours[i]));
			}
		}



		RNG rng(12345);
		/// Draw contours + rotated rects + ellipses
		Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
		for (int i = 0; i< contours.size(); i++)
		{
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			// contour
			drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
			// ellipse
			//ellipse(drawing, minRect[i], color, 2, 8);
			// rotated rectangle
			Point2f rect_points[4]; minRect[i].points(rect_points);
			for (int j = 0; j < 4; j++)
				line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		}

		/// Show in a window
		namedWindow("Contours", CV_WINDOW_AUTOSIZE);
		imshow("Contours", drawing);


	}

	waitKey();
}