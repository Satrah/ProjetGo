#include <iostream>
#include "ImageLoader.h"

using namespace std;

int main()
{
	const char* imgFilename = "../Images/IMG_3589.JPG";
    GOProject::ImageLoader loader;
	if (!loader.Load(imgFilename))
		cerr << "Unable to load image file " << imgFilename << endl;
	else
		loader.DebugDisplay();
    return 0;
}