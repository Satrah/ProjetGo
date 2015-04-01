#include "ImageLoader.h"
#include "Algo.h"

#include <opencv2/highgui/highgui.hpp>

using namespace GOProject;
using namespace cv;

void AlGo::charge(ImageLoader loader)
{
	_taillePlateau = loader.GetSize();
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
			_plateau[BoardPosition(i, j)] = CASE_VIDE;
}
void AlGo::refresh(ImageLoader loader)
{
	Image<uchar> I = loader.GetImage();
	Image<uchar> J;
	cornerHarris(I, J, 2, 3, 0.04);
	GaussianBlur(J, J, Size(3,3), 0,0);
	imshow("Test", J);
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
		{
			Point ici = Point((i*I.height()) / _taillePlateau, (j*I.height()) / _taillePlateau);
			if (I(ici) > 150)
				_plateau[BoardPosition(i, j)] = CASE_BLANCHE;
			else if (J(ici) > 5);
			else 
				_plateau[BoardPosition(i, j)] = CASE_NOIRE;
			
			switch (_plateau[BoardPosition(i, j)])
			{
			case CASE_BLANCHE:
				circle(I, ici, 10, Scalar(200), 10); break;
			case CASE_NOIRE:
				circle(I, ici, 10, Scalar(0), 10); break;
			default:
				break;
			}
		}
	imshow("Ronds", I);
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