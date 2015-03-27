#include <cassert>
#include <cstdlib>
#include <math.h>       /* fmod */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "ImageLoader.h"
#include "Utils.h"

using namespace cv;
using namespace GOProject;


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
	// Update mask
	_globalRectangleMask = Image<uchar>(Mat::zeros(_loadedImage.size(), CV_8UC1));
	fillRectangle(_globalRectangleMask, _globalRectangle);
	for (int j = 0; j < 4; j++)
		line(drawing, rect_points[j], rect_points[(j + 1) % 4], Scalar(255, 255, 255), 1, 8);

	imshow("Corner2", drawing);
}


void ImageLoader::TrackFeaturesInsideBoard()
{
	// Prepare image
	Image<uchar> imageForTracking = _loadedImage.clone();
	equalizeHist(_loadedImage, imageForTracking, _globalRectangleMask);
	// Detect what we should track
	vector <Point2f> corners;
	goodFeaturesToTrack(imageForTracking, corners, TRACKING_NUM_POINTS, TRACKING_QUALITY, TRACKING_MIN_DIST, _globalRectangleMask, 3, false);
	// Debug display
	Image<uchar> displayDebug = imageForTracking.clone();
	for (auto& point : corners)
		circle(displayDebug, point, 5, Scalar(0), 3);
	imshow("Tracking", displayDebug);
}
void ImageLoader::TrackFeaturesInsideBoard2()
{
	// Calculate mask
	if (_noeuds.empty())
	{
		Image<uchar> mask = Mat::zeros(_loadedImage.size(), CV_8UC1);
		fillRectangle(mask, _globalRectangle);
		// Prepare image
		_imageForTracking = _loadedImage.clone();
		equalizeHist(_loadedImage, _imageForTracking, mask);
		// Detect what we should track
		goodFeaturesToTrack(_imageForTracking, _noeuds, TRACKING_NUM_POINTS, TRACKING_QUALITY, TRACKING_MIN_DIST, mask, 3, true);
	}
	else
	{
		std::vector<Point2f> noeud;
		Image<uchar> mask = Mat::zeros(_loadedImage.size(), CV_8UC1);
		fillRectangle(mask, _globalRectangle);
		int assez = 0;
		for (int i = 0; i < _noeuds.size(); i++)
		{
			Image<uchar> mask2 = Mat::zeros(_loadedImage.size(), CV_8UC1);
			fillRectangle(mask2, RotatedRect(_noeuds[i], Size(10, 10), 0));
			goodFeaturesToTrack(_loadedImage, noeud, 1, TRACKING_QUALITY, TRACKING_MIN_DIST, mask2, 3, true);
			if (!noeud.empty())
			{
				_noeuds[i] = noeud[0];
				assez++;
			}
			else _noeuds[i] = cv::Point2f(30, 30);
		}
		_imageForTracking = _loadedImage.clone();
		equalizeHist(_loadedImage, _imageForTracking, mask);
		Image<uchar> displayDebug = _imageForTracking.clone();
		for (auto& point : _noeuds)
			circle(displayDebug, point, 5, Scalar(0), 3);
		imshow("Tracking", displayDebug);
		if (assez < 6)
			_noeuds.clear();
	}
	// Debug display
	/*Image<uchar> displayDebug = imageForTracking.clone();
	for (auto& point : corners)
	circle(displayDebug, point, 5, Scalar(0), 3);
	imshow("Tracking", displayDebug);*/
}

bool ImageLoader::FindHomographyWithDetectedRectangles()
{
	// TODO: Fix this copy-paste
	if (_detectedRectangles.size() <= 4 || !_nbCases)
		return false;
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
	// 
	Point2f unityX(float(cos(_rectangleOrientation)), float(sin(_rectangleOrientation)));
	Point2f unityY(float(sin(_rectangleOrientation)), -float(cos(_rectangleOrientation)));
	Point2f topLeft(_topLeft.x, _topLeft.y);
	_homographyCurrentFrame = _homographyCurrentFrame % RECTANGLE_HOMOGRAPHY_FRAMES_MEMORY;
	std::vector<Point2f>& points1 = _homographyOriginalPoints[_homographyCurrentFrame];
	std::vector<Point2f>& points2 = _homographyTransformedPoints[_homographyCurrentFrame];
	points1.clear();
	points2.clear();

	int distCasesPixels = *(_loadedImage.size) / _nbCases;
	points1.push_back(Point2f(topLeft));
	points2.push_back(Point2f(distCasesPixels / 2, distCasesPixels / 2));
	for (auto& rectangle : _detectedRectangles)
	{
		int coordX = -round((rectangle.center - topLeft).dot(unityX) / medianw);
		int coordY = -round((rectangle.center - topLeft).dot(unityY) / medianh);
		if (coordX >= 0 && coordX < _nbCases &&
			coordY >= 0 && coordY < _nbCases)
		{
			points1.push_back(Point2f(rectangle.center));
			points2.push_back(Point2f(distCasesPixels / 2 + coordX * distCasesPixels, distCasesPixels / 2 + coordY * distCasesPixels));
		}
	}
	// Merge with previous points
	std::vector<Point2f> allPoints1;
	std::vector<Point2f> allPoints2;
	for (int i = 0; i < RECTANGLE_HOMOGRAPHY_FRAMES_MEMORY; ++i)
	{
		allPoints1.insert(allPoints1.end(), _homographyOriginalPoints[i].begin(), _homographyOriginalPoints[i].end());
		allPoints2.insert(allPoints2.end(), _homographyTransformedPoints[i].begin(), _homographyTransformedPoints[i].end());
	}
	// Not enough points to find an homography
	if (allPoints1.size() < 10)
		return false;
	_homography = findHomography(allPoints1, allPoints2, CV_RANSAC);
	return true;
}

void ImageLoader::DetectIntersect()
{
	if (_detectedRectangles.size() <= 1)
		return;
	vector<double> heights;
	vector<double> widths;
	for (auto& rectangle : _detectedRectangles)
	{
		heights.push_back(rectangle.size.height);
		widths.push_back(rectangle.size.width);
	}
	sort(heights.begin(), heights.end());
	sort(widths.begin(), widths.end());
	double medianh = heights[heights.size() / 2];
	double medianw = widths[widths.size() / 2];
	int nbSquaresw = ceil(_globalRectangle.size.width / medianw - 0.2);
	int nbSquaresh = ceil(_globalRectangle.size.height / medianh - 0.2);

	if (nbSquaresw == nbSquaresh)
		_nbCasesTab[(++_currentCase) % TRACKING_NB_IMAGES_FOR_CASES_COUNT] = nbSquaresw;
	else
		_nbCasesTab[(++_currentCase) % TRACKING_NB_IMAGES_FOR_CASES_COUNT] = std::max(nbSquaresw, nbSquaresh);

	std::map<int, int> nbCasesProba;
	for (int i = 0; i < TRACKING_NB_IMAGES_FOR_CASES_COUNT; ++i)
		nbCasesProba[_nbCasesTab[i]]++;
	int max = 0;
	int ind = -1;
	for (auto& i : nbCasesProba)
	if (i.second > max)
	{
		max = i.second;
		ind = i.first;
	}
	_nbCases = ind;
}
