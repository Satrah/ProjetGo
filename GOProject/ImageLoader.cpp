#include <cassert>
#include <cstdlib>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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

void ImageLoader::DetectLinesHough(int threshold, int minLineLength, int maxLineGap)
{
	if (!threshold)
		return;
	Image<uchar> dst;
	Canny(GetImage(), dst, 50, 200, 3);
	_houghLines.clear();
	HoughLinesP(dst, _houghLines, 1, CV_PI / 100, threshold, minLineLength, maxLineGap);
}

Image<uchar> ImageLoader::DisplayHoughLines(const char* winName) const
{
	//Image<uchar> cdst = GetImage();
	Image<uchar> cdst(GetImage().width(), GetImage().height(), CV_8UC1);
	for (size_t i = 0; i < _houghLines.size(); i++)
	{
		Vec4i l = _houghLines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
	}
	imshow(winName, cdst);
	return cdst;
}

void ImageLoader::BuildHoughLinesHistogram()
{
	for (int i = 0; i < HOUGH_LINES_HISTO_ORIG_COUNT; ++i)
		_houghLinesOrigHistogram[i] = 0.0f;
	if (!_houghLines.size())
		return;
	for (size_t i = 0; i < _houghLines.size(); i++)
	{
		Vec4i l = _houghLines[i];
		double angle = atan(float(l[2] - l[0]) / (l[3] - l[1]));
		angle += CV_PI / 2;
		int idx = int(floor(angle / CV_PI * HOUGH_LINES_HISTO_ORIG_COUNT));
		if (idx >= HOUGH_LINES_HISTO_ORIG_COUNT)
			idx = HOUGH_LINES_HISTO_ORIG_COUNT - 1;
		_houghLinesOrigHistogram[idx]++;
	}
	// Normalize
	double max = 0.0f;
	for (size_t i = 0; i < HOUGH_LINES_HISTO_ORIG_COUNT; i++)
	if (_houghLinesOrigHistogram[i] > max)
	{
		max = _houghLinesOrigHistogram[i];
		_houghLinesMaxDirection = i * CV_PI / HOUGH_LINES_HISTO_ORIG_COUNT;
	}
	for (size_t i = 0; i < HOUGH_LINES_HISTO_ORIG_COUNT; i++)
		_houghLinesOrigHistogram[i] /= max;
}


int comparInt(void* a, void* b)
{
	if (*((int*)a) < *((int*)b))
		return 1;
	else if (*((int*)a) == *((int*)b))
		return 0;
	return -1;
}
void ImageLoader::FilterHoughLines(int nIterations)
{
	std::vector<Vec4i> verticalLines;
	std::vector<Vec4i> horizontalLines;
	for (size_t i = 0; i < _houghLines.size(); i++)
	{
		Vec4i& l = _houghLines[i];
		float angle = atan(float(l[2] - l[0]) / (l[3] - l[1]));
		if (fabs(angle - _houghLinesMaxDirection) < CV_PI / 4)
			verticalLines.push_back(l);
		else
			horizontalLines.push_back(l);
	}
	if (!verticalLines.size() || !horizontalLines.size())
		return;
	for (int i = 0; i < nIterations; ++i)
	{
		Vec4i& randomVertical = verticalLines[rand() % verticalLines.size()];
		Vec4i& randomHorizontal = horizontalLines[rand() % horizontalLines.size()];
		std::vector<Point2f> points1;
		points1.push_back(Point2f(randomVertical[0], randomVertical[1]));
		points1.push_back(Point2f(randomVertical[2], randomVertical[3]));
		std::vector<Point2f> points2;
		points2.push_back(Point2f(randomVertical[0], randomVertical[1]));
		points2.push_back(Point2f(randomVertical[2], randomVertical[3]));
		Mat H = findHomography(points1, points2, CV_RANSAC);
	}
}

void ImageLoader::DisplayTransformedImage() const
{

}

void ImageLoader::DisplayHoughLinesOrientation(const char* winName) const
{
	const int histogramLineWidth = 10;
	const int heighPrecision = 100;
	Image<uchar> histogram(HOUGH_LINES_HISTO_ORIG_COUNT*histogramLineWidth, heighPrecision, CV_8UC1);
	// Display results
	for (size_t i = 0; i < HOUGH_LINES_HISTO_ORIG_COUNT; i++)
	{
		int count = int(_houghLinesOrigHistogram[i] * heighPrecision);
		line(histogram, Point(i*histogramLineWidth, 0), Point(i*histogramLineWidth, count), Scalar(0, 0, 255), histogramLineWidth, CV_AA);
	}
	imshow(winName, histogram);
}

void ImageLoader::DebugDisplay()
{
	imshow("GO Image Loader display", _loadedImage);
	waitKey();
}