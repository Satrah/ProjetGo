#include <cassert>
#include <cstdlib>
#include <math.h>       /* fmod */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ImageLoader.h"
#include "Utils.h"

using namespace cv;
using namespace GOProject;

const double ImageLoader::TRACKING_QUALITY = 0.01;
const double ImageLoader::TRACKING_MIN_DIST = 20;

bool ImageLoader::Load(const char* filename)
{
	_loadedImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	if (!_loadedImage.data)
		return false;
	_globalRectangleMask = Image<uchar>(Mat::zeros(_loadedImage.size(), CV_8UC1));
	return true;
}

void ImageLoader::Detect()
{
	assert(Loaded());
}

void ImageLoader::DisplayTransformedImage() const
{
	if (_homography.empty())
		return;
	Mat wrappedImage = GetImage().clone();
	warpPerspective(GetImage(), wrappedImage, _homography, wrappedImage.size());
	imshow("PerspectiveTransform", wrappedImage);
}
