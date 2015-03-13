#include <cassert>
#include <cstdlib>
#include <math.h>       /* fmod */

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
// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2,
	Point2f &r)
{
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
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
	return out / sqrt(out.dot(out));
}

inline void transformLinesAndNormalize(LinesVec const& lines, std::vector<Vec2f>& out, Mat const& H)
{
	for (LinesVec::const_iterator it = lines.begin(); it != lines.end(); ++it)
		out.push_back(transformVectorAndNormalize(*it, H));
}

void ImageLoader::FilterHoughLines(int nIterations, int nSuccessfullIterations)
{
	LinesVec verticalLines;
	LinesVec horizontalLines;
	for (size_t i = 0; i < _houghLines.size(); i++)
	{
		Vec4i& l = _houghLines[i];
		float angle = atan(float(l[2] - l[0]) / (l[3] - l[1]));
		angle = fmod(angle - _houghLinesMaxDirection, CV_PI);
		if (fabs(angle) < CV_PI / 4 || fabs(angle - CV_PI) < CV_PI / 4)
			verticalLines.push_back(l);
		else
			horizontalLines.push_back(l);
	}
	if (!verticalLines.size() || !horizontalLines.size())
		return;
	int bestHscore = 0;
	int successFullIterations = 0;
	for (int i = 0; i < nIterations || successFullIterations < nSuccessfullIterations; ++i)
	{
		/// First, try to compute a random homography from 4 lines
		Vec4i& randomVertical = verticalLines[rand() % verticalLines.size()];
		Vec4i& randomVertical2 = verticalLines[rand() % verticalLines.size()];
		Vec4i& randomHorizontal = horizontalLines[rand() % horizontalLines.size()];
		Vec4i& randomHorizontal2 = horizontalLines[rand() % horizontalLines.size()];
		std::vector<Point2f> points1;
		if (!AppendIntersec(randomVertical, randomHorizontal, points1) ||
			!AppendIntersec(randomVertical2, randomHorizontal, points1) ||
			!AppendIntersec(randomVertical, randomHorizontal2, points1) ||
			!AppendIntersec(randomVertical2, randomHorizontal2, points1))
			continue;
		std::vector<Point2f> points2;
		const unsigned int COORD_BASIS = 100;
		const float SCALE = 100;
		points2.push_back(Point2f(COORD_BASIS, COORD_BASIS));
		points2.push_back(Point2f(COORD_BASIS + SCALE, COORD_BASIS));
		points2.push_back(Point2f(COORD_BASIS, COORD_BASIS + SCALE));
		points2.push_back(Point2f(COORD_BASIS + SCALE, COORD_BASIS + SCALE));
		Mat H = findHomography(points1, points2, CV_RANSAC);
		++successFullIterations;
		// Now, calculate the score of this homography: Other lines should also be normal.
		int currentScore = 0;
		/* First method: exhaustive check O(n²)
		std::vector<Vec2f> vertTransform;
		transformLinesAndNormalize(verticalLines, vertTransform, H);
		std::vector<Vec2f> horizontalTransform;
		transformLinesAndNormalize(horizontalLines, horizontalTransform, H);
		for (std::vector<Vec2f>::const_iterator vert = vertTransform.begin(); vert != vertTransform.end(); ++vert)
			for (std::vector<Vec2f>::const_iterator horiz = horizontalTransform.begin(); horiz != horizontalTransform.end(); ++horiz)
				if (fabs(vert->dot(*horiz)) < 0.1f)
					++currentScore;
					*/
		/* Second */
		std::vector<Vec2f> vertTransform;
		transformLinesAndNormalize(verticalLines, vertTransform, H);
		std::vector<Vec2f> horizontalTransform;
		transformLinesAndNormalize(horizontalLines, horizontalTransform, H);
		for (std::vector<Vec2f>::const_iterator vert = vertTransform.begin(); vert != vertTransform.end(); ++vert)
			if (fabs((*vert)[0]) < 0.1f)
				++currentScore;
		for (std::vector<Vec2f>::const_iterator horiz = horizontalTransform.begin(); horiz != horizontalTransform.end(); ++horiz)
			if (fabs((*horiz)[1]) < 0.1f)
				++currentScore;
		/* Third possible method: monte carlo.
		for (int i = 0; i < 50; ++i)
		{
			Vec2f vert = transformVectorAndNormalize(verticalLines[rand() % verticalLines.size()], H);
			Vec2f horiz = transformVectorAndNormalize(horizontalLines[rand() % horizontalLines.size()], H);
			double scalarProd = vert.dot(horiz);
			if (fabs(scalarProd) < 0.1f)
				++currentScore;
		} */
		if (currentScore > bestHscore)
		{
			bestHscore = currentScore;
			_homography = H;
		}
	}
}

void ImageLoader::DisplayTransformedImage() const
{
	if (_homography.empty())
		return;
	Mat wrappedImage = GetImage().clone();
	warpPerspective(GetImage(), wrappedImage, _homography, wrappedImage.size());
	imshow("PerspectiveTransform", wrappedImage);
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