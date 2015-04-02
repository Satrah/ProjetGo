#include <cassert>
#include <cstdlib>
#include <math.h>       /* fmod */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ImageLoader.h"
#include "Utils.h"

using namespace cv;
using namespace GOProject;

void ImageLoader::DetectLinesHough(int threshold, int minLineLength, int maxLineGap)
{
	if (!threshold)
		return;
	Image<uchar> dst;
	Canny(GetImage(), dst, 50, 200, 3);
	for (int i = 0; i < dst.width(); ++i)
	for (int j = 0; j < dst.height(); ++j)
		dst(i, j) = dst(i, j)*_globalRectangleMask(i, j);

	_houghLines.clear();
	HoughLinesP(dst, _houghLines, 1, CV_PI / 100, threshold, minLineLength, maxLineGap);
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
		double angle = atan(double(l[2] - l[0]) / (l[3] - l[1]));
		angle += CV_PI / 2;
		int idx = int(floor(angle / CV_PI * HOUGH_LINES_HISTO_ORIG_COUNT));
		if (idx >= HOUGH_LINES_HISTO_ORIG_COUNT)
			idx = HOUGH_LINES_HISTO_ORIG_COUNT - 1;
		_houghLinesOrigHistogram[idx]++;
	}
	// Normalize
	double max = 0.0f;
	for (size_t i = 0; i < HOUGH_LINES_HISTO_ORIG_COUNT; ++i)
	if (_houghLinesOrigHistogram[i] > max)
	{
		max = _houghLinesOrigHistogram[i];
		_houghLinesMaxDirection = i * CV_PI / HOUGH_LINES_HISTO_ORIG_COUNT;
	}
	for (size_t i = 0; i < HOUGH_LINES_HISTO_ORIG_COUNT; ++i)
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

inline bool AppendIntersec(Vec4i& a, Vec4i& b, std::vector<Point2f>& intersec)
{
	Point2f out;
	if (intersection(Point2f(a[0], a[1]), Point2f(a[2], a[3]), Point2f(b[0], b[1]), Point2f(b[2], b[3]), out))
	{
		intersec.push_back(out);
		return true;
	}
	return false;
}

inline Vec2f transformVectorAndNormalize(Vec4i point, Mat H)
{
	std::vector<Vec2f> orig;
	orig.push_back(Point2f(point[0], point[1]));
	orig.push_back(Point2f(point[2], point[3]));
	std::vector<Vec2f> transformed = orig;
	perspectiveTransform(orig, transformed, H);
	Vec2f out(transformed[1][0] - transformed[0][0], transformed[1][1] - transformed[0][1]);
	out /= sqrt(out.dot(out));
	return out;
}

inline void transformLinesAndNormalize(LinesVec const& lines, std::vector<Vec2f>& out, Mat const& H)
{
	for (LinesVec::const_iterator it = lines.begin(); it != lines.end(); ++it)
		out.push_back(transformVectorAndNormalize(*it, H));
}

void ImageLoader::FilterVerticalLines()
{
	_verticalLines.clear();
	_horizontalLines.clear();
	for (size_t i = 0; i < _houghLines.size(); i++)
	{
		Vec4i& l = _houghLines[i];
		double angle = atan(double(l[2] - l[0]) / (l[3] - l[1]));
		angle += CV_PI / 2;
		angle = fmod(angle - _houghLinesMaxDirection, CV_PI);
		if (fabs(angle) < CV_PI / 4 || fabs(angle - CV_PI) < CV_PI / 4)
			_verticalLines.push_back(l);
		else
			_horizontalLines.push_back(l);
	}
}
void ImageLoader::FindBestHomography(int nIterations, int nSuccessfullIterations)
{
	if (!_verticalLines.size() || !_horizontalLines.size())
	{
		if (!_verticalLines.size())
			printf("/!\\ No vertical lines!\n");
		if (!_horizontalLines.size())
			printf("/!\\ No horizontal lines!\n");
		return;
	}
	int bestHscore = 0;
	int successFullIterations = 0;
	for (int i = 0; i < nIterations || successFullIterations < nSuccessfullIterations; ++i)
	{
		/// First, try to compute a random homography from 4 lines
		Vec4i& randomVertical = _verticalLines[rand() % _verticalLines.size()];
		Vec4i& randomVertical2 = _verticalLines[rand() % _verticalLines.size()];
		Vec4i& randomHorizontal = _horizontalLines[rand() % _horizontalLines.size()];
		Vec4i& randomHorizontal2 = _horizontalLines[rand() % _horizontalLines.size()];
		std::vector<Point2f> points1;
		if (!AppendIntersec(randomVertical, randomHorizontal, points1) ||
			!AppendIntersec(randomVertical2, randomHorizontal, points1) ||
			!AppendIntersec(randomVertical, randomHorizontal2, points1) ||
			!AppendIntersec(randomVertical2, randomHorizontal2, points1))
			continue;

		std::vector<Point2f> points2;
		const float COORD_BASIS = points1[0].x;
		const float SCALE = sqrt((points1[0] - points1[1]).dot(points1[0] - points1[1]));
		points2.push_back(Point2f(COORD_BASIS, COORD_BASIS));
		points2.push_back(Point2f(COORD_BASIS + SCALE, COORD_BASIS));
		points2.push_back(Point2f(COORD_BASIS, COORD_BASIS + SCALE));
		points2.push_back(Point2f(COORD_BASIS + SCALE, COORD_BASIS + SCALE));
		Mat H = findHomography(points1, points2, CV_RANSAC);
		++successFullIterations;
		// Now, calculate the score of this homography.
		int currentScore = 0;
		/* First method: exhaustive check O(n²)
		std::vector<Vec2f> vertTransform;
		transformLinesAndNormalize(_verticalLines, vertTransform, H);
		std::vector<Vec2f> horizontalTransform;
		transformLinesAndNormalize(_horizontalLines, horizontalTransform, H);
		for (std::vector<Vec2f>::const_iterator vert = vertTransform.begin(); vert != vertTransform.end(); ++vert)
		for (std::vector<Vec2f>::const_iterator horiz = horizontalTransform.begin(); horiz != horizontalTransform.end(); ++horiz)
		if (fabs(vert->dot(*horiz)) < 0.1f)
		++currentScore;
		/* Second */
		std::vector<Vec2f> vertTransform;
		transformLinesAndNormalize(_verticalLines, vertTransform, H);
		std::vector<Vec2f> horizontalTransform;
		transformLinesAndNormalize(_horizontalLines, horizontalTransform, H);
		for (std::vector<Vec2f>::const_iterator vert = vertTransform.begin(); vert != vertTransform.end(); ++vert)
		if (fabs((*vert)[0]) < 0.1f)
			++currentScore;
		for (std::vector<Vec2f>::const_iterator horiz = horizontalTransform.begin(); horiz != horizontalTransform.end(); ++horiz)
			if (fabs((*horiz)[1]) < 0.1f)
				++currentScore;
		if (currentScore > bestHscore)
		{
			bestHscore = currentScore;
			_homography = H;
		}
	}
	if (!bestHscore)
		printf("/!\\ No homography found!\n");
}

void ImageLoader::ApplyHomography()
{
	warpPerspective(GetImage(), GetImage(), _homography, GetImage().size());
}
void ImageLoader::ClearBadLines()
{
	std::vector<Vec2f> vertTransform;
	transformLinesAndNormalize(_verticalLines, vertTransform, _homography);
	std::vector<Vec2f> horizontalTransform;
	transformLinesAndNormalize(_horizontalLines, horizontalTransform, _homography);
	int shift = 0;
	for (size_t i = 0; i < vertTransform.size(); ++i)
	if (fabs(vertTransform[i][0]) > 0.04f)
	{
		_verticalLines.erase(_verticalLines.begin() + i - shift);
		++shift;
	}
	shift = 0;
	for (size_t i = 0; i < horizontalTransform.size(); ++i)
	if (fabs(horizontalTransform[i][1]) > 0.04f)
	{
		_horizontalLines.erase(_horizontalLines.begin() + i - shift);
		++shift;
	}
}


void ImageLoader::DisplayHoughLinesOrientation(const char* winName) const
{
	const int histogramLineWidth = 10;
	const int heighPrecision = 100;
	Image<uchar> histogram(HOUGH_LINES_HISTO_ORIG_COUNT*histogramLineWidth, heighPrecision, CV_8UC1);
	for (uchar* i = histogram.datastart; i < histogram.dataend; ++i)
		*i = 255;
	// Display results
	for (size_t i = 0; i < HOUGH_LINES_HISTO_ORIG_COUNT; i++)
	{
		int count = int(_houghLinesOrigHistogram[i] * heighPrecision);
		line(histogram, Point(i*histogramLineWidth, 0), Point(i*histogramLineWidth, count), Scalar(0, 0, 255), histogramLineWidth, CV_AA);
	}
	imshow(winName, histogram);
}

Image<uchar> ImageLoader::DisplayHoughLines(const char* winName) const
{
	Image<uchar> cdst(GetImage().width(), GetImage().height(), CV_8UC1);
	for (uchar* i = cdst.datastart; i < cdst.dataend; ++i)
		*i = 255;
	for (size_t i = 0; i < _houghLines.size(); ++i)
	{
		Vec4i l = _houghLines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
	}
	imshow(winName, cdst);
	return cdst;
}

void ImageLoader::DisplayVerticalAndHorizontalLines(const char* winName)
{
	Mat cdst(GetImage().height(), GetImage().width(), CV_8UC3);
	for (uchar* i = cdst.datastart; i < cdst.dataend; ++i)
		*i = 0;
	for (size_t i = 0; i < _houghLines.size(); ++i)
	{
		Vec4i l = _houghLines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(100, 100, 100), 1, CV_AA);
	}
	for (size_t i = 0; i < _verticalLines.size(); ++i)
	{
		Vec4i l = _verticalLines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 1, CV_AA);
	}
	for (size_t i = 0; i < _horizontalLines.size(); ++i)
	{
		Vec4i l = _horizontalLines[i];
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 255, 0), 1, CV_AA);
	}
	imshow(winName, cdst);
}
