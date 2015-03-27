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
		ImageLoader() : _homographyCurrentFrame(0) {
			for (int i = 0; i < TRACKING_NB_IMAGES_FOR_CASES_COUNT; ++i)
				_nbCasesTab[i] = 0;
		}

		bool Load(const char* imageFile);
		inline bool Load(Image<uchar> image) { _loadedImage = image; return Loaded(); }
		inline bool Loaded() const { return _loadedImage.data != NULL; }
		void Detect();
		// Hough lines to find an homography
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
		void TrackFeaturesInsideBoard2();
		bool FindHomographyWithDetectedRectangles();
		void DetectIntersect();
		inline int GetSize(){return _nbCases};
	protected:
		void MoveLine(cv::Point& begin, cv::Point2f const& direction);
	public:
		void DebugDisplaySquares() const;

		inline Image<uchar> GetImage() const { return _loadedImage; }
		static const int TRACKING_NB_IMAGES_FOR_CASES_COUNT = 20;
		static const int HOUGH_LINES_HISTO_ORIG_COUNT = 10;
		static const int TRACKING_NUM_POINTS = 60;
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
		Image<uchar> _globalRectangleMask;
		cv::Point _topLeft;
		cv::Point _botRight;
		int _boardSize;
		int _nbCases = 0;
		int _currentCase = 0;
		int _nbCasesTab[TRACKING_NB_IMAGES_FOR_CASES_COUNT];
		std::vector<cv::Point2f> _noeuds;
		Image<uchar> _imageForTracking;

		const static int RECTANGLE_HOMOGRAPHY_FRAMES_MEMORY = 10;
		int _homographyCurrentFrame;
		std::vector<cv::Point2f> _homographyOriginalPoints[RECTANGLE_HOMOGRAPHY_FRAMES_MEMORY];
		std::vector<cv::Point2f> _homographyTransformedPoints[RECTANGLE_HOMOGRAPHY_FRAMES_MEMORY];
	};
};

#endif