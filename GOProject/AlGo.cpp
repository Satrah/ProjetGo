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
int AlGo::_calculLibAux(int i, int j, std::map<BoardPosition, bool> calcule)
{
	BoardPosition ici = BoardPosition(i, j);
	EtatCase couleur = _plateau[ici];
	if (calcule[ici])
		return _libertes[ici];
	calcule[ici] = true;
	if (i > 0 && _plateau[BoardPosition(i - 1, j)] == couleur)
	{
		if (!calcule[BoardPosition(i - 1, j)])
			_libertes[ici] += _calculLibAux(i - 1, j, calcule);
		else
			_libertes[ici] += _libertes[BoardPosition(i - 1, j)];
	}
	if (i < _taillePlateau && _plateau[BoardPosition(i + 1, j)] == couleur)
	{
		if (!calcule[BoardPosition(i + 1, j)])
			_libertes[ici] += _calculLibAux(i + 1, j, calcule);
		else
			_libertes[ici] += _libertes[BoardPosition(i + 1, j)];
	}
	if (j > 0 && _plateau[BoardPosition(i, j - 1)] == couleur)
	{
		if (!calcule[BoardPosition(i, j - 1)])
			_libertes[ici] += _calculLibAux(i, j - 1, calcule);
		else
			_libertes[ici] += _libertes[BoardPosition(i, j - 1)];
	}
	if (j < _taillePlateau && _plateau[BoardPosition(i, j + 1)] == couleur)
	{
		if (!calcule[BoardPosition(i, j + 1)])
			_libertes[ici] += _calculLibAux(i, j + 1, calcule);
		else
			_libertes[ici] += _libertes[BoardPosition(i, j + 1)];
	}
	return _libertes[ici];
}
void AlGo::calculLibertes()
{
	std::map<BoardPosition, bool> calcule;
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
		{
			calcule[BoardPosition(i, j)] = false;
		}
	//calcule des cases vides adjacentes
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
		{
			BoardPosition ici = BoardPosition(i, j);
			EtatCase couleur = _plateau[ici];
			_libertes[ici] = 0;
			if (couleur = CASE_VIDE)
				continue;
			if (_plateau[BoardPosition(i - 1, j)] == CASE_VIDE)
				_libertes[ici]++;
		}
	//calcule par groupe
	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
		{
			_libertes[BoardPosition(i, j)] = _calculLibAux(i, j, calcule);
		}
}