#ifndef _HEADER_IMAGE_LOADER
#define _HEADER_IMAGE_LOADER

#include "Image.h"

namespace GOProject
{
	class ImageLoader
	{
	public:
		ImageLoader() {}
		bool Load(const char* imageFile);
		void DebugDisplay();
	protected:
		Image<uchar> _loadedImage;
	};
};

#endif