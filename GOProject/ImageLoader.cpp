#include <opencv2/highgui/highgui.hpp>
#include "ImageLoader.h"

using namespace cv;
using namespace GOProject;

bool ImageLoader::Load(const char* filename)
{
	_loadedImage = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	return _loadedImage.data;
}

void ImageLoader::DebugDisplay()
{
	imshow("GO Image Loader display", _loadedImage);
}