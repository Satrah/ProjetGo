#ifndef _HEADER_IMAGE_LOADER
#define _HEADER_IMAGE_LOADER

#include <opencv2/imgproc/imgproc.hpp>
#include "Image.h"

namespace GOProject
{
	class ImageLoader
	{
	public:
		ImageLoader() {}

		bool Load(const char* imageFile);
		inline bool Loaded() const { return _loadedImage.data != NULL; }
		void Detect();

		void DebugDisplay();
	protected:
		Image<uchar> _loadedImage;
	};
};

#endif