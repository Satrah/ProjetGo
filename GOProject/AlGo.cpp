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
	_pxlPerCase = loader.GetImage().size().height / (_taillePlateau + 1);
}
void AlGo::refresh(ImageLoader const& loader)
{
	Image<float> harrisCorners;
	//Image<uchar> K;
	cornerHarris(loader.GetImage(), harrisCorners, 17, 7, 0.04);
	//GaussianBlur(J, J, Size(13, 13), 5, 5);
	Image<uchar> imageBlurred;
	GaussianBlur(loader.GetImage(), imageBlurred, Size(7, 7), 4, 4);
	_memory[_current] = harrisCorners;
	_current++;
	_current %= MEMORY_FRAMES;

	for (int i = 0; i < _taillePlateau; ++i)
		for (int j = 0; j < _taillePlateau; ++j)
		{
			Point ici = Point(((i+1)*imageBlurred.height()) / (_taillePlateau + 1), ((j+1)*imageBlurred.height()) / (_taillePlateau + 1));
			float sum = 0;
			for (int k = 0; k < _memory.size(); ++k)
				if (!_memory[k].empty())
					sum += _memory[k](ici);
			sum /= _memory.size();
			EtatCase& currCase = _plateau[BoardPosition(i, j)];
			if (sum > 1.0f)
				currCase = CASE_VIDE;
			else if (imageBlurred(ici) > 100)
				currCase = CASE_BLANCHE;
			else 
				currCase = CASE_NOIRE;
			
			switch (currCase)
			{
			case CASE_BLANCHE:
				circle(imageBlurred, ici, 10, Scalar(200), 10);
				break;
			case CASE_NOIRE:
				circle(imageBlurred, ici, 10, Scalar(0), 10);
				break;
			default:
				//circle(I, ici, 8, Scalar(200), 3);
				break;
			}
		}
	imshow("imageBlurred", imageBlurred);
	imshow("harrisCorners", harrisCorners);
}

void AlGo::affichePlateau()
{
	if (_plateau.empty())
		return;
	const static int CASE_SIZE = 30; // Pixels
	const static int BOARD_MARGIN = 30;
	Image<Vec3b> cdst(2 * BOARD_MARGIN + _taillePlateau * CASE_SIZE, 2 * BOARD_MARGIN + _taillePlateau * CASE_SIZE, CV_8UC3);
	for (uchar* i = cdst.datastart; i < cdst.dataend; ++i)
		*i = 200;
	Scalar white(230, 230, 230);
	Scalar black(0, 0, 0);
	// Vertical lines
	for (int x = 0; x < _taillePlateau; ++x)
		line(cdst, Point(BOARD_MARGIN + x*CASE_SIZE, BOARD_MARGIN), Point(BOARD_MARGIN + x*CASE_SIZE, BOARD_MARGIN + (_taillePlateau-1)*CASE_SIZE), Scalar(0, 0, 0), 2);
	// Horizontal lines
	for (int y = 0; y < _taillePlateau; ++y)
		line(cdst, Point(BOARD_MARGIN, BOARD_MARGIN + y*CASE_SIZE), Point(BOARD_MARGIN + (_taillePlateau-1)*CASE_SIZE, BOARD_MARGIN + y*CASE_SIZE), Scalar(0, 0, 0), 2);
	// Points
	for (int x = 0; x < (_taillePlateau + 1); ++x)
		for (int y = 0; y < (_taillePlateau + 1); ++y)
			switch (_plateau[BoardPosition(x, y)])
			{
			case CASE_NOIRE:
			case CASE_BLANCHE:
				circle(cdst, Point(BOARD_MARGIN + x*CASE_SIZE, BOARD_MARGIN + y*CASE_SIZE), 4, _plateau[BoardPosition(x, y)] == CASE_NOIRE ? black : white, 10);
				circle(cdst, Point(BOARD_MARGIN + x*CASE_SIZE, BOARD_MARGIN + y*CASE_SIZE), 8, _plateau[BoardPosition(x, y)] == CASE_NOIRE ? black * 0.3 : black * 0.8, 2);
				break;
			}
	imshow("Plateau", cdst);
}

bool AlGo::render(Image<cv::Vec4b>& out)
{
	int w = _taillePlateau + 1;
	// Draw lines
	for (int x = 0; x < _taillePlateau; ++x)
	{
		line(out, Point((x + 1)*_pxlPerCase, _pxlPerCase), Point((x + 1)*_pxlPerCase, _taillePlateau*_pxlPerCase), Scalar(0, 0, 0, 255), 2);
		line(out, Point(_pxlPerCase, (x + 1)*_pxlPerCase), Point(_taillePlateau*_pxlPerCase, (x + 1)*_pxlPerCase), Scalar(0, 0, 0, 255), 2);
	}
	if (_areas)
	{
		for (int x = 0; x < w; ++x)
		for (int y = 0; y < w; ++y)
		{
			int area = _areas[y*w + x];
			if (area && _ownerByArea[area] != (CASE_NOIRE | CASE_BLANCHE | CASE_VIDE))
			{
				Scalar color = _ownerByArea[area] & CASE_NOIRE ? Scalar(0, 0, 0, 150) : Scalar(255, 255, 255, 150);
				circle(out, Point((x + 1) * _pxlPerCase, (y + 1) * _pxlPerCase), _pxlPerCase / 3, color, 10);
			}
		}
	}
	return true;
}

int AlGo::_calculLibAux(int i, int j, std::map<BoardPosition, bool>& calcule)
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
			if (couleur == CASE_VIDE)
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

void AlGo::computeAreas()
{
	if (_areas)
		delete[] _areas;
	int w = _taillePlateau + 1;
	int size = w*w;
	_areas = new int[size];
	for (int x = 0; x < w; ++x)
		for (int y = 0; y < w; ++y)
			_areas[w*y + x] = GetCase(x, y) == CASE_VIDE ? w*x + y + 1 : 0;
	bool changed = true;
	while (changed)
	{
		changed = false;
		for (int x = 0; x < w; ++x)
		for (int y = 0; y < w; ++y)
		{
			int& caseArea = _areas[w*y+x];
			if (!caseArea) // =0 s'il y a un pion ici
				continue;
			for (int i = -1; i <= 1; ++i)
			for (int j = -1; j <= 1; ++j)
			{
				int x2 = x + i;
				int y2 = y + j;
				if (x2 < w && x2 >= 0 && y2 < w && y2 >= 0 && _areas[w*y2 + x2] > caseArea)
				{
					changed = true;
					caseArea = _areas[w*y2 + x2];
				}
			}
		}
	}
}
void AlGo::computeBWAreas()
{
	_ownerByArea.clear();
	int w = _taillePlateau + 1;
	for (int x = 0; x < w; ++x)
	for (int y = 0; y < w; ++y)
	{
		_ownerByArea[_areas[y*w + x]] = CASE_VIDE;
	}
	for (int x = 0; x < w; ++x)
	for (int y = 0; y < w; ++y)
	{
		int& caseArea = _areas[w*y + x];
		if (!caseArea) // =0 s'il y a un pion ici
			continue;
		for (int i = -1; i <= 1; ++i)
		for (int j = -1; j <= 1; ++j)
		{
			int x2 = x + i;
			int y2 = y + j;
			if (x2 < w && x2 >= 0 && y2 < w && y2 >= 0)
			{
				_ownerByArea[caseArea] = EtatCase(_ownerByArea[caseArea] | GetCase(x2, y2));
			}
		}
	}
}