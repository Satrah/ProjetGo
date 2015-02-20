#include <cassert>

#include <opencv2/highgui/highgui.hpp>

#include "ImageLoader.h"

using namespace cv;
using namespace GOProject;

bool ImageLoader::Load(const char* filename)
{
	_loadedImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	return _loadedImage.data != NULL;
}

void ImageLoader::Detect()
{
	assert(Loaded());
}

void ImageLoader::DetectLinesHough()
{
	Mat dst;
	Canny(GetImage(), dst, 50, 200, 3);
	_houghLines.clear();
	HoughLinesP(dst, _houghLines, 1, CV_PI / 180, 50, 50, 10);
}
void ImageLoader::DisplayHoughLines(const char* winName) const
{
	Mat cdst = GetImage();
	for (size_t i = 0; i < _houghLines.size(); i++)
	{
		Vec4i l = _houghLines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
	}
	imshow(winName, cdst);
}
void ImageLoader::DebugDisplay()
{
	imshow("GO Image Loader display", _loadedImage);
	waitKey();
}