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
			CASE_VIDE		= 'v',
			CASE_NOIRE		= 'n',
			CASE_BLANCHE	= 'b',
		};
		AlGo() : _pxlPerCase(0) { Image<float> I;  for (int i = 0; i < MEMORY_FRAMES; ++i) _memory.push_back(I); }
		void charge(ImageLoader const& loader);
		void refresh(ImageLoader const& loader);
		bool render(Image<cv::Vec4b>& out);
		void affichePlateau();
		void calculLibertes();
	protected:
		static const int MEMORY_FRAMES = 40;
		typedef std::pair<int, int> BoardPosition;
		int _calculLibAux(int i, int j, std::map<BoardPosition, bool> calcule);
		std::map<BoardPosition, EtatCase> _plateau;
		std::map<BoardPosition, int> _libertes;
		int _taillePlateau;
		std::vector<Image<float>> _memory;
		int _current = 0;
		int _pxlPerCase;
	};
}