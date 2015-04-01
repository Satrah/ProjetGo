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
		void charge(ImageLoader loader);
		void refresh(ImageLoader loader);
		void suggereCoup(ImageLoader loader);
		void affichePlateau();
	protected:
		typedef std::pair<int, int> BoardPosition;
		std::map<BoardPosition, EtatCase> _plateau;
		int _taillePlateau;
	};
}