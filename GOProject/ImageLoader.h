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
		// Hough lines
		void DetectLinesHough(int threshold = 50, int minLineLength = 50, int maxLineGap = 10);
		Image<uchar> DisplayHoughLines(const char* winName = "hough lines") const;
		void BuildHoughLinesHistogram();
		void DisplayHoughLinesOrientation(const char* winName = "hough lines orig") const;
		void FilterHoughLines(int nIterations = 1000, int nSuccessfullIterations = 10);
		void DisplayTransformedImage() const;

		void DebugDisplay();
		inline Image<uchar> GetImage() const { return _loadedImage; }

		static const int HOUGH_LINES_HISTO_ORIG_COUNT = 10;
	protected:
		Image<uchar> _loadedImage;
		LinesVec _houghLines;
		cv::Mat _homography;
		double _houghLinesOrigHistogram[HOUGH_LINES_HISTO_ORIG_COUNT];
		double _houghLinesMaxDirection;
	};
};

#endif