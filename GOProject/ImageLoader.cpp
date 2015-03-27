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
		/* Third possible method: monte carlo.
		for (int i = 0; i < 50; ++i)
		{
			Vec2f vert = transformVectorAndNormalize(_verticalLines[rand() % _verticalLines.size()], H);
			Vec2f horiz = transformVectorAndNormalize(_horizontalLines[rand() % _horizontalLines.size()], H);
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


void ImageLoader::DetectSquareForms()
{
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	/// Detect edges using Threshold
	threshold(GetImage(), threshold_output, 100, 255, THRESH_BINARY);
	/// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	/// Find the rotated rectangles and ellipses for each contour
	_detectedRectangles.clear();
	_detectedRectangles.reserve(contours.size());

	// Find at the same time the square median area
	vector<double> squareAreas;
	vector<double> squareOrientation;
	for (int i = 0; i < contours.size(); i++)
	{
		RotatedRect rec = minAreaRect(Mat(contours[i]));
		// Filter out too small rectangles
		double area = rec.size.area();
		if (area < 20 * 20)
			continue;
		// Also remove rectangles that are not squares.
		if (fabs((rec.size.height - rec.size.width) / (rec.size.height + rec.size.width)) > 0.2f)
			continue;
		squareAreas.push_back(area);
		_detectedRectangles.push_back(rec);
	}
	if (_detectedRectangles.empty())
		return;
	// Filter our squares not in the median area
	// TODO: Do that in a linear time
	sort(squareAreas.begin(), squareAreas.end());
	double median = squareAreas[squareAreas.size() / 2];
	for (int i = 0; i < _detectedRectangles.size();)
	{
		double area = _detectedRectangles[i].size.area();
		if (fabs((area - median) / median) > 0.2f)
			_detectedRectangles.erase(_detectedRectangles.begin() + i);
		else
		{
			squareOrientation.push_back(_detectedRectangles[i].angle);
			++i;
		}
	}
	if (!squareOrientation.empty())
	{
		sort(squareOrientation.begin(), squareOrientation.end());
		_rectangleOrientation = squareOrientation[squareOrientation.size() / 2];
		// Deg to rad
		_rectangleOrientation = _rectangleOrientation * CV_PI / 180;
	}
}

void ImageLoader::DebugDisplaySquares() const
{
	RNG rng(123456);
	/// Draw contours + rotated rects + ellipses
	Mat drawing = Mat::zeros(GetImage().size(), CV_8UC3);
	for (int i = 0; i < _detectedRectangles.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Point2f rect_points[4];
		_detectedRectangles[i].points(rect_points);
		for (int j = 0; j < 4; j++)
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
	}
	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
	Point2f rect_points[4];
	_globalRectangle.points(rect_points);
	for (int j = 0; j < 4; j++)
		line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 4, 8);
	/// Show in a window
	imshow("Squares", drawing);
}

void ImageLoader::DetectBoard1()
{
	int maxScore = -1;
	for (int i = 0; i < _detectedRectangles.size(); ++i)
		for (int j = 0; j < _detectedRectangles.size(); ++j)
		{
			Point2f pointsRectI[4], pointsRectJ[4];
			_detectedRectangles[i].points(pointsRectI);
			_detectedRectangles[j].points(pointsRectJ);
			Point2f a = pointsRectI[0];
			Point2f b = pointsRectJ[3];
			double rectSizeX = (a - b).dot(Point2f(cos(_rectangleOrientation), sin(_rectangleOrientation)));
			double rectSizeY = (a - b).dot(Point2f(sin(_rectangleOrientation), cos(_rectangleOrientation)));
			RotatedRect rec = RotatedRect((a + b) * 0.5, Size(fabs(rectSizeX), fabs(rectSizeY)), _rectangleOrientation * 180 / CV_PI);
			Point2f contour[4];
			rec.points(contour);
			std::vector<Point> contourVec;
			for (int l = 0; l < 4; ++l)
				contourVec.push_back(Point(contour[l].x, contour[l].y));
			int insideCount = 0;
			for (int l = 0; l < _detectedRectangles.size(); ++l)
			{
				RotatedRect& rec2 = _detectedRectangles[l];
				double ret = pointPolygonTest(contourVec, rec2.center, false);
				if (ret > 0)
					++insideCount;
			}
			//insideCount *= 1 / (fabs(rec.angle - _rectangleOrientation) + 1);
			if (insideCount > maxScore)
			{
				_globalRectangle = rec;
				maxScore = insideCount;
			}
		}
		printf("Final rec with score %u  orig=%f\n", maxScore, _rectangleOrientation);
}

double distanceToLine(cv::Point2f line_start, cv::Point2f line_end, cv::Point point)
{
	double normalLength = _hypot(line_end.x - line_start.x, line_end.y - line_start.y);
	double distance = (double)((point.x - line_start.x) * (line_end.y - line_start.y) - (point.y - line_start.y) * (line_end.x - line_start.x)) / normalLength;
	return distance;
}

void ImageLoader::MoveLine(Point& begin, Point2f const& direction)
{
	if (_detectedRectangles.empty())
		return;
	struct RectanglesSorter
	{
		Point2f lineA;
		Point2f lineB;
		bool operator() (RotatedRect& i, RotatedRect& j)
		{
			return distanceToLine(lineA, lineB, i.center) < distanceToLine(lineA, lineB, j.center);
		}
		inline void MovePoint(Point& point, Point2f const& direc, RotatedRect const& rect)
		{
			double dist = distanceToLine(lineA, lineB, rect.center);
			point = point - Point(dist * direc);
		}
	};
	Point2f lineSecondPoint = Point2f(begin) + Point2f(direction.y, -direction.x);
	vector<RotatedRect> rectanglesList = _detectedRectangles;
	RectanglesSorter sorter;
	sorter.lineA = Point2f(begin);
	sorter.lineB = lineSecondPoint;
	int pos = 0;
	if (rectanglesList.size() > 5)
		pos = 1;
	std::nth_element(rectanglesList.begin(), rectanglesList.begin() + pos, rectanglesList.end(), sorter);
	sorter.MovePoint(begin, direction, rectanglesList[pos]);
	//printf("Pt moved to [%i %i] center is [%f %f]\n", begin.x, begin.y, rectanglesList[0].center.x, rectanglesList[0].center.y);
}

void ImageLoader::DetectBoard2()
{
	if (_detectedRectangles.size() <= 1)
		return;
	Point top, bot, left, right;
	Point2f topToBot(float(sin(_rectangleOrientation)), -float(cos(_rectangleOrientation)));
	Point2f leftToRight(float(cos(_rectangleOrientation)), float(sin(_rectangleOrientation)));
	MoveLine(bot, -topToBot);
	MoveLine(top, topToBot);
	MoveLine(left, leftToRight);
	MoveLine(right, -leftToRight);
	// Find corners now
	Point2f tf, br;
	if (!intersection(top, Point2f(top) + leftToRight, left, Point2f(left) - topToBot, tf) ||
		!intersection(bot, Point2f(bot) + leftToRight, right, Point2f(right) - topToBot, br))
		return;
	// ... And create the rectangle :)
	_globalRectangle = RotatedRect((tf + br) * 0.5, Size(fabs((tf - br).dot(leftToRight)), fabs((tf - br).dot(topToBot))), _rectangleOrientation * 180 / CV_PI);
	_topLeft = Point(tf);
	_botRight = Point(br);
	// Size of the board ?
	_boardSize = 0;

	// Debug display:
	RNG rng(123456);
	Mat drawing = Mat::zeros(GetImage().size(), CV_8UC3);
	for (int i = 0; i < _detectedRectangles.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		Point2f rect_points[4];
		_detectedRectangles[i].points(rect_points);
		for (int j = 0; j < 4; j++)
			line(drawing, rect_points[j], rect_points[(j + 1) % 4], color, 1, 8);
		line(drawing, _detectedRectangles[i].center - 10 * topToBot, _detectedRectangles[i].center + 10 * topToBot, color, 2, 8);
		line(drawing, _detectedRectangles[i].center - 10 * leftToRight, _detectedRectangles[i].center + 10 * leftToRight, color, 2, 8);
	}
	line(drawing, Point2f(top) - 1000 * leftToRight, Point2f(top) + 1000 * leftToRight, Scalar(0, 0, 120), 5, 8);
	line(drawing, Point2f(bot) - 1000 * leftToRight, Point2f(bot) + 1000 * leftToRight, Scalar(120, 0, 120), 4, 8);
	line(drawing, Point2f(left) - 1000 * topToBot, Point2f(left) + 1000 * topToBot, Scalar(0, 120, 120), 3, 8);
	line(drawing, Point2f(right) - 1000 * topToBot, Point2f(right) + 1000 * topToBot, Scalar(120, 120, 120), 2, 8);
	Point2f rect_points[4];
	_globalRectangle.points(rect_points);
	for (int j = 0; j < 4; j++)
		line(drawing, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 255, 255), 1, 8);
	
	imshow("Corner2", drawing);
}


void ImageLoader::TrackFeaturesInsideBoard()
{
	// Calculate mask
	Image<uchar> mask = Mat::zeros(_loadedImage.size(), CV_8UC1);
	fillRectangle(mask, _globalRectangle);
	// Prepare image
	Image<uchar> imageForTracking = _loadedImage.clone();
	equalizeHist(_loadedImage, imageForTracking, mask);
	// Detect what we should track
	vector <Point2f> corners;
	goodFeaturesToTrack(imageForTracking, corners, TRACKING_NUM_POINTS, TRACKING_QUALITY, TRACKING_MIN_DIST, mask, 3, true);
	// Debug display
	Image<uchar> displayDebug = imageForTracking.clone();
	for (auto& point : corners)
		circle(displayDebug, point, 5, Scalar(0), 3);
	imshow("Tracking", displayDebug);
}

void ImageLoader::DetectIntersect()
{
	if (_detectedRectangles.size() <= 1)
		return;
	vector<double> heights;
	vector<double> widths;
	for (int i = 0; i < _detectedRectangles.size(); i++)
	{
		heights.push_back(_detectedRectangles[i].size.height);
		widths.push_back(_detectedRectangles[i].size.width);
	}
	sort(heights.begin(), heights.end());
	sort(widths.begin(), widths.end());
	double medianh = heights[heights.size() / 2];
	double medianw = widths[widths.size() / 2];
	int nbSquaresw = ceil(_globalRectangle.size.width / medianw -0.1);
	int nbSquaresh = ceil(_globalRectangle.size.height / medianh -0.1);
	if (nbSquaresw == nbSquaresh)
	{
		_nbCasesTab[(++_currentCase) % TRACKING_NB_IMAGES_FOR_CASES_COUNT] = nbSquaresw;
		printf("ok : %d\n", nbSquaresw);
	}
	else
	{	
		printf("Non : %d ou %d ?\n", nbSquaresw, nbSquaresh);
		_nbCasesTab[(++_currentCase) % TRACKING_NB_IMAGES_FOR_CASES_COUNT] = std::max(nbSquaresw, nbSquaresh);
	}
	int nbCasesProba[20];
	for (int i = 0; i < 20; ++i)
		nbCasesProba[i] = 0;
	for (int i = 0; i < TRACKING_NB_IMAGES_FOR_CASES_COUNT; ++i)
		nbCasesProba[_nbCasesTab[i]]++;
	int max = 0;
	int ind = -1;
	for (int i = 0; i < 20; ++i)
		if (nbCasesProba[i] > max)
		{
			max = nbCasesProba[i];
			ind = i;
		}
	_nbCases = ind;
	printf("Nombre de cases probable : %d", ind);

}
