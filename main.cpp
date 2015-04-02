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
	VideoCapture cap; // Will capture 8UC3
	cap.open(0);
	Mat webcam;
	Image<uchar> webcamGrey;
	int cornersOffset = 50;
	GOProject::ImageLoader loader;
	GOProject::PerspectiveFinder perspectiveFinder(cornersOffset);
	int consecutiveSuccess = 0;
	// Calibrate
	while (consecutiveSuccess < 5)
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
		//imshow("HomographyTransformed", perspectiveFinder);
	}
	GOProject::AlGo go;
	go.charge(loader);
	cv::Mat homographyInv = perspectiveFinder.GetHomography().inv() * loader.GetHomography().inv();
	while (true)
	{
		char k = waitKey(50);
		if (k == 'q')
			return;
		if (k == 'p') // Pause
			continue;
		cap >> webcam;
		if (webcam.empty())
			continue;
		cvtColor(webcam, webcamGrey, CV_BGR2GRAY);
		// 1- Apply homographies to get the board points at specific pixels
		perspectiveFinder.Load(webcamGrey);
		perspectiveFinder.HomographyTransform();
		loader.Load(perspectiveFinder);
		loader.ApplyHomography();

		// 2- Handle the data
		go.refresh(loader);
		go.affichePlateau();
		int h = webcam.size().height;
		Image<Vec4b> rendered = Mat::zeros(h, h, CV_8UC4);
		if (go.render(rendered))
		{
			cv::Mat output = rendered.clone();

			// 3- Apply the inversed homography to display augmented reality go !
			warpPerspective(rendered, output, homographyInv, rendered.size());
			for (int x = 0; x < h; ++x)
			for (int y = 0; y < h; ++y)
			{
				Vec3b& p = webcam.at<Vec3b>(x, y);
				Vec4b& renderedPoint = output.at<Vec4b>(x, y);
				if (renderedPoint[3] > 0)
					for (int i = 0; i < 3; ++i)
						p[i] = (p[i] * (255 - renderedPoint[3]) + renderedPoint[i] * renderedPoint[3]) / 255;
			}
			imshow("Output", webcam);
		}
	}
}