#ifndef _HEADER_IMAGE_LOADER
#define _HEADER_IMAGE_LOADER

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <vector>

#include "Image.h"

namespace GOProject
{
	typedef std::vector<cv::Vec4i> LinesVec;
	class ImageLoader
	{
	public:
		ImageLoader() {}

		bool Load(const char* imageFile);
		inline bool Load(Image<uchar> image) { _loadedImage = image; return Loaded(); }
		inline bool Loaded() const { return _loadedImage.data != NULL; }
		void Detect();
		// Hough lines and homography finding
		void DetectLinesHough(int threshold = 50, int minLineLength = 50, int maxLineGap = 10);
		Image<uchar> DisplayHoughLines(const char* winName = "hough lines") const;
		void DisplayVerticalAndHorizontalLines(const char* winName = "hough lines cleared");
		void BuildHoughLinesHistogram();
		void DisplayHoughLinesOrientation(const char* winName = "hough lines orig") const;
		void FilterVerticalLines();
		void FindBestHomography(int nIterations = 1000, int nSuccessfullIterations = 250);
		void ApplyHomography();
		void ClearBadLines();
		void DisplayTransformedImage() const;
		// Rectangle detection
		void DetectSquareForms();
		void DetectBoard1();
		void DetectBoard2();
		void TrackFeaturesInsideBoard();
		void DetectIntersect();
	protected:
		void MoveLine(cv::Point& begin, cv::Point2f const& direction);
	public:
		void DebugDisplaySquares() const;

		inline Image<uchar> GetImage() const { return _loadedImage; }

		static const int HOUGH_LINES_HISTO_ORIG_COUNT = 10;
		static const int TRACKING_NUM_POINTS = 15;
		static const double TRACKING_QUALITY;
		static const double TRACKING_MIN_DIST;
	protected:
		Image<uchar> _loadedImage;

		LinesVec _verticalLines;
		LinesVec _horizontalLines;
		LinesVec _houghLines;
		cv::Mat _homography;
		double _houghLinesOrigHistogram[HOUGH_LINES_HISTO_ORIG_COUNT];
		double _houghLinesMaxDirection;
		double _rectangleOrientation;

		std::vector<cv::RotatedRect> _detectedRectangles;
		cv::RotatedRect _globalRectangle;
		cv::Point _topLeft;
		cv::Point _botRight;
		int _boardSize;
		int _nbCases = 0;
	};
};

#endif