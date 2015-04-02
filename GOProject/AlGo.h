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
		void charge(ImageLoader const& loader);
		void refresh(ImageLoader const& loader);
		bool render(Image<cv::Vec4b>& out);
		void affichePlateau();
		AlGo() : _pxlPerCase(0) { Image<float> I;  for (int i = 0; i < MEMORY_FRAMES; ++i) _memory.push_back(I); }
	protected:
		static const int MEMORY_FRAMES = 40;
		typedef std::pair<int, int> BoardPosition;
		std::map<BoardPosition, EtatCase> _plateau;
		int _taillePlateau;
		std::vector<Image<float>> _memory;
		int _current = 0;
		int _pxlPerCase;
	};
}