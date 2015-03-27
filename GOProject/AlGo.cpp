#include "ImageLoader.h"
#include "Algo.h"

using namespace GOProject;
using namespace cv;

void AlGo::charge(ImageLoader loader)
{
	_taillePlateau = loader.GetSize();
	for (int i = 0; i < _taillePlateau; ++i)
		for (j = 0; j < _taillePlateau; ++j)
			_plateau[Point(i, j)] = CASE_VIDE;
}