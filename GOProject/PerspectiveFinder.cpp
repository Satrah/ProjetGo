#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "PerspectiveFinder.h"
#include "Utils.h"

using namespace GOProject;
using namespace cv;

/**
 * We assume that we have horizontal and vertical lines for the board.
 */
bool PerspectiveFinder::HomographyCalibrate()
{
	const float MAXIMUM_HORIZONTAL_DEVIATION = 0.1f;
	const float MAXIMUM_VERTICAL_DEVIATION = 0.5f;
	const unsigned int VERTICAL_LINES_FIRST_BOTTOM = unsigned int(0.8f * height()); // Vertical lines start in the 20% bottom of the image
	const unsigned int VERTICAL_LINES_LAST_TOP = unsigned int(0.5f * height());
	const float SHIFT_CORNERS_OFFSET = 0.2f;

	/// 1- Process image with Canny
	Image<uchar> cannyImg;
	Canny(*this, cannyImg, 50, 200, 3);

	/// 2- Detect Hough lines from Canny
	vector<vector<cv::Point> > lineSegments;
	vector<cv::Point> aux;
	LinesVec houghLines;
	LinesVec verticalLines, horizontalLines;
	// threshold, minLineLength, maxLineGap
	HoughLinesP(cannyImg, houghLines, 1, CV_PI / 180, 40, 50, 30);

	/// 3- Filter vertical / horizontal lines
	Image<Vec3b> cdst(width(), height(), CV_8UC3);
	for (uchar* i = cdst.datastart; i < cdst.dataend; ++i)
		*i = 200;
	int minXBottom = width();
	int maxXBottom = 0;
	for (auto& l : houghLines)
	{
		bool vertical = false;
		bool horizontal = false;
		double angle = atan(double(l[2] - l[0]) / (l[3] - l[1]));
		if (fabs(angle - CV_PI / 2) < MAXIMUM_HORIZONTAL_DEVIATION || fabs(angle + CV_PI / 2) < MAXIMUM_HORIZONTAL_DEVIATION)
		{
			for (unsigned int i = 0; i < 2; ++i)
				if (l[2*i+1] >= VERTICAL_LINES_FIRST_BOTTOM)
				{
					if (l[2 * i] < minXBottom)
						minXBottom = l[2 * i];
					if (l[2 * i] > maxXBottom)
						maxXBottom = l[2 * i];
				}
			horizontalLines.push_back(l);
			horizontal = true;
		}
		else if (fabs(angle) < MAXIMUM_VERTICAL_DEVIATION || fabs(angle + CV_PI) < MAXIMUM_VERTICAL_DEVIATION)
		{
			if (l[1] < l[3])
			{
				std::swap(l[0], l[2]);
				std::swap(l[1], l[3]);
			}
			// l[0..1] = bottom
			// l[2..3] = top
			// so l[1] > l[3]
			if (l[1] >= VERTICAL_LINES_FIRST_BOTTOM
				&& l[3] <= VERTICAL_LINES_LAST_TOP)
			{
				verticalLines.push_back(l);
				vertical = true;
			}
		}
		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(100, horizontal ? 200 : 100, vertical ? 200 : 100), 1, CV_AA);
	}
	line(cdst, Point(0, VERTICAL_LINES_FIRST_BOTTOM), Point(width(), VERTICAL_LINES_FIRST_BOTTOM), Scalar(0, 0, 0), 2, CV_AA);
	line(cdst, Point(0, VERTICAL_LINES_LAST_TOP), Point(width(), VERTICAL_LINES_LAST_TOP), Scalar(0, 0, 0), 2, CV_AA);
	line(cdst, Point(minXBottom, VERTICAL_LINES_FIRST_BOTTOM), Point(minXBottom, height()), Scalar(0, 0, 0), 2, CV_AA);
	line(cdst, Point(maxXBottom, VERTICAL_LINES_FIRST_BOTTOM), Point(maxXBottom, height()), Scalar(0, 0, 0), 2, CV_AA);

	/// 4- How to find good points for an homography ?
	// Find top left / top right / bot left / bot right from the vertical lines
	Vec4i leftLine;
	Vec4i rightLine;
	double maxAngle = -MAXIMUM_HORIZONTAL_DEVIATION - 0.1f;
	double minAngle = MAXIMUM_HORIZONTAL_DEVIATION + 0.1f;
	minXBottom -= 20;
	maxXBottom += 20;
	for (auto& l : verticalLines)
	{
		if (l[0] < minXBottom || l[0] > maxXBottom)
			continue;
		double angle = atan(double(l[2] - l[0]) / (l[3] - l[1]));
		if (angle < minAngle)
		{
			minAngle = angle;
			leftLine = l;
		}
		if (angle > maxAngle)
		{
			maxAngle = angle;
			rightLine = l;
		}
	}

	// 5- We're done ! We have a left line, a right line, let's compute the divine homography NOW
	// But first, are our left / right line valid ?
	float line1H = abs(leftLine[1] - leftLine[3]);
	float line2H = abs(rightLine[1] - rightLine[3]);
	bool pointsValid = true;
	// Diff height ?
	if (fabs((line1H - line2H) / (line1H + line2H)) > 0.2f)
		pointsValid = false;
	if (fabs((leftLine[0] - minXBottom) / float(width())) > 0.05f)
		pointsValid = false;
	if (fabs((rightLine[0] - maxXBottom) / float(width())) > 0.05f)
		pointsValid = false;

	if (!pointsValid && _pointsOrig.size())
	{
		// Invalid
		line(cdst, Point(leftLine[0], leftLine[1]), Point(leftLine[2], leftLine[3]), Scalar(0, 255, 0), 2, CV_AA);
		line(cdst, Point(rightLine[0], rightLine[1]), Point(rightLine[2], rightLine[3]), Scalar(0, 255, 0), 2, CV_AA);
	}
	else
	{
		line(cdst, Point(leftLine[0], leftLine[1]), Point(leftLine[2], leftLine[3]), Scalar(0, 0, 255), 2, CV_AA);
		line(cdst, Point(rightLine[0], rightLine[1]), Point(rightLine[2], rightLine[3]), Scalar(0, 0, 255), 2, CV_AA);

		if (_pointsOrig.size() > _memoryPoints)
		{
			_pointsOrig.erase(_pointsOrig.begin(), _pointsOrig.begin() + 4);
			_pointsDest.erase(_pointsDest.begin(), _pointsDest.begin() + 4);
		}
		float coordMin = height() * SHIFT_CORNERS_OFFSET;
		float coordMax = height() - coordMin;
		_pointsOrig.push_back(Point2f(coordMin, coordMin));
		_pointsDest.push_back(Point2f(rightLine[2], rightLine[3]));
		_pointsOrig.push_back(Point2f(coordMin, coordMax));
		_pointsDest.push_back(Point2f(rightLine[0], rightLine[1]));
		_pointsOrig.push_back(Point2f(coordMax, coordMin));
		_pointsDest.push_back(Point2f(leftLine[2], leftLine[3]));
		_pointsOrig.push_back(Point2f(coordMax, coordMax));
		_pointsDest.push_back(Point2f(leftLine[0], leftLine[1]));
	}
	_homography = findHomography(_pointsDest, _pointsOrig, CV_RANSAC);
	imshow("Hough Lines", cdst);
	return true;
}

void PerspectiveFinder::HomographyTransform()
{
	Image<uchar> me = clone();
	warpPerspective(me, *this, _homography, size());
}