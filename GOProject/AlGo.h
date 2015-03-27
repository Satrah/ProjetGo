#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImageLoader.h"

namespace GOProject
{
	class AlGo {
	public:
		enum EtatCase
		{
			CASE_VIDE,
			CASE_NOIRE,
			CASE_BLANCHE,
		};
		void charge(ImageLoader loader);
		void suggereCoup(ImageLoader loader);
	protected:
		std::map<cv::Point, EtatCase> _plateau;
		int _taillePlateau;
	};
}