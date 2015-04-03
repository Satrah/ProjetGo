#pragma once

#include <vector>
#include <map>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageLoader.h"

namespace GOProject
{
	class AlGo {
	public:
		enum EtatCase
		{
			CASE_VIDE			= 0x01,
			CASE_NOIRE			= 0x02,
			CASE_BLANCHE		= 0x03,
		};
		AlGo() : _pxlPerCase(0), _areas(0) { Image<float> I;  for (int i = 0; i < MEMORY_FRAMES; ++i) _memory.push_back(I); }
		void charge(ImageLoader const& loader);
		void refresh(ImageLoader const& loader);
		bool render(Image<cv::Vec4b>& out);
		void affichePlateau();
		void calculLibertes();
		void computeAreas();
	protected:
		static const int MEMORY_FRAMES = 40;
		typedef std::pair<int, int> BoardPosition;
		int _calculLibAux(int i, int j, std::map<BoardPosition, bool>& calcule);
		EtatCase& GetCase(int x, int y) { return _plateau[BoardPosition(x, y)];  }

		std::map<BoardPosition, EtatCase> _plateau;
		std::map<BoardPosition, int> _libertes;
		int* _areas;
		int _taillePlateau;
		std::vector<Image<float>> _memory;
		int _current = 0;
		int _pxlPerCase;
	};
}