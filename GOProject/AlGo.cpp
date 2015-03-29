#include "ImageLoader.h"
#include "Algo.h"

using namespace GOProject;
using namespace cv;

void AlGo::charge(ImageLoader loader)
{
	_taillePlateau = loader.GetSize();
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
			_plateau[BoardPosition(i, j)] = CASE_VIDE;
}

void AlGo::affichePlateau()
{
	if (_plateau.empty())
		return;
	printf("\n");
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
		{
			printf("%c", _plateau[BoardPosition(i, j)]);
			if (j + 1 == _taillePlateau)
				printf("\n");
		}
}
void AlGo::suggereCoup(ImageLoader loader)
{
	//circle(loader.GetImage(), Point(100, 100), 50, 10, 3);
}