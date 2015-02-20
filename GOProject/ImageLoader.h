#ifndef _HEADER_IMAGE_LOADER
#define _HEADER_IMAGE_LOADER

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <vector>

#include "Image.h"

namespace GOProject
{
	class ImageLoader
	{
	public:
		ImageLoader() {}

		bool Load(const char* imageFile);
		inline bool Load(Image<uchar> image) { _loadedImage = image; return Loaded(); }
		inline bool Loaded() const { return _loadedImage.data != NULL; }
		void Detect();
		void DetectLinesHough();
		void DisplayHoughLines(const char* winName = "hough lines") const;

		void DebugDisplay();
		inline Image<uchar> GetImage() const { return _loadedImage; }
	protected:
		Image<uchar> _loadedImage;
		std::vector<cv::Vec4i> _houghLines;
	};
};

#endif