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

VideoCapture cap;
void GOGameLaunch();

int main()
{
	cap.open(0);
	GOGameLaunch();
    return 0;
}

cv::Mat CalibrateGame(GOProject::PerspectiveFinder& perspectiveFinder, GOProject::ImageLoader& loader)
{
	Mat webcam;
	Image<uchar> webcamGrey;
	int consecutiveSuccess = 0;
	// Calibrate
	while (consecutiveSuccess < 20)
	{
		char k = waitKey(10);
		if (k == 'q')
			return cv::Mat();
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
		loader.DetectSquareForms(perspectiveFinder.GetCornersOffset(), webcamGrey.height() - perspectiveFinder.GetCornersOffset(), perspectiveFinder.GetCornersOffset(), webcamGrey.height() - perspectiveFinder.GetCornersOffset());
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
	return perspectiveFinder.GetHomography().inv() * loader.GetHomography().inv();
}

void GOGameLaunch()
{
	Mat webcam;
	Image<uchar> webcamGrey;
	int cornersOffset = 50;
	GOProject::ImageLoader loader;
	GOProject::PerspectiveFinder perspectiveFinder(cornersOffset);
	GOProject::AlGo go;
	go.charge(loader);
	cv::Mat homographyInv = CalibrateGame(perspectiveFinder, loader);
	printf("Got homography %u\n", homographyInv.empty());
	while (true)
	{
		char k = waitKey(50);
		if (k == 'q' || homographyInv.empty())
		{
			printf("Got Q\n");
			return;
		}
		if (k == 'c')
		{
			homographyInv = CalibrateGame(perspectiveFinder, loader);
			continue;
		}
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