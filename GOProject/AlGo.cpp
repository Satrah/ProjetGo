#include "ImageLoader.h"
#include "Algo.h"

#include <opencv2/highgui/highgui.hpp>

using namespace GOProject;
using namespace cv;

void AlGo::charge(ImageLoader const& loader)
{
	_taillePlateau = loader.GetSize()+1;
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
			_plateau[BoardPosition(i, j)] = CASE_VIDE;
}
void AlGo::refresh(ImageLoader const& loader)
{
	Image<uchar> I = loader.GetImage().clone();
	Image<uchar> J;
	//Image<uchar> K;
	cornerHarris(I, J, 7, 7, 0.04);
	//GaussianBlur(J, J, Size(3, 3), 5, 5);
	//GaussianBlur(I, K, Size(3, 3), 4, 4);
	imshow("Test", J);
	_memory[_current] = J;
	_current++;
	_current %= MEMORY_FRAMES;

	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
		{
			Point ici = Point((i*I.height()) / (_taillePlateau-1), (j*I.height()) / (_taillePlateau-1));
			printf("%d ", J(ici));
			if (j + 1 == _taillePlateau)
				printf("\n");
			int sum = 0;
			for (int k = 0; k < _memory.size(); ++k)
				if (!_memory[k].empty())
					sum += _memory[k](ici);
			sum /= _memory.size();
			if (sum > 100)
				_plateau[BoardPosition(i, j)] = CASE_VIDE;
			else if (I(ici) > 100)
				_plateau[BoardPosition(i, j)] = CASE_BLANCHE;
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
void AlGo::suggereCoup(ImageLoader const& loader)
{
	//circle(loader.GetImage(), Point(100, 100), 50, 10, 3);
}