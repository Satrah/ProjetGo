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
		void suggereCoup(ImageLoader const& loader);
		void affichePlateau();
		void calculLibertes();
		AlGo(){ Image<uchar> I;  for (int i = 0; i < MEMORY_FRAMES; ++i) _memory.push_back(I); }
	protected:
		static const int MEMORY_FRAMES = 40;
		typedef std::pair<int, int> BoardPosition;
		int _calculLibAux(int i, int j, std::map<BoardPosition, bool> calcule);
		std::map<BoardPosition, EtatCase> _plateau;
		std::map<BoardPosition, int> _libertes;
		int _taillePlateau;
		std::vector<Image<uchar>> _memory;
		int _current = 0;
	};
}