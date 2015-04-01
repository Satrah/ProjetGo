#include <cassert>
#include <cstdlib>
#include <math.h>       /* fmod */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ImageLoader.h"
#include "Utils.h"

using namespace cv;
using namespace GOProject;


void ImageLoader::DetectEllipse()
{
	if (_loadedImage.empty())
		return;
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);

	/// Detect edges using Threshold
	threshold(_loadedImage, threshold_output, 100, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the rotated rectangles and ellipses for each contour
	vector<RotatedRect> minEllipse(contours.size());

	for (auto& contour: contours)
	{
		if (contour.size() > 5)
		{
			RotatedRect e = fitEllipse(Mat(contour));
			if (e.size.width < 100 && e.size.height < 100 /*&& e.size.width > 5 &&  e.size.height > 5*/)
				minEllipse.push_back(e);
		}
	}

	/// Draw ellipses
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for (int i = 0; i< minEllipse.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		// ellipse
		ellipse(drawing, minEllipse[i], color, 2, 8);
	}

	/// Show in a window
	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
}