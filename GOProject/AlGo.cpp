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
			/*
			printf("%3d ", J(ici));
			if (j + 1 == _taillePlateau)
				printf("\n");*/
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
	// Draw lines
	for (int x = 0; x < _taillePlateau; ++x)
	{
		line(out, Point((x + 1)*_pxlPerCase, _pxlPerCase), Point((x + 1)*_pxlPerCase, _taillePlateau*_pxlPerCase), Scalar(0, 0, 0, 255), 2);
		line(out, Point(_pxlPerCase, (x + 1)*_pxlPerCase), Point(_taillePlateau*_pxlPerCase, (x + 1)*_pxlPerCase), Scalar(0, 0, 0, 255), 2);
	}
	circle(out, Point(3 * _pxlPerCase, 3 * _pxlPerCase), _pxlPerCase/2, Scalar(153, 76, 42, 255), 10);
	return true;
}