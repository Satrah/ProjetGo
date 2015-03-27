#ifndef _HEADER_OPENCV_IMAGE_WRAPPER
#define _HEADER_OPENCV_IMAGE_WRAPPER

#include <vector>
#include <ImageLoader.h>
#include <opencv2/imgproc/imgproc.hpp>

namespace GOProject
{
	template <typename T>
	class AlGo {
	public:
		enum EtatCase
		{
			CASE_VIDE,
			CASE_NOIRE,
			CASE_BLANCHE,
		};
		void charge(ImageLoader loader);

	protected:
		std::map<cv::Point, EtatCase> _plateau;
		int _taillePlateau;
	};
}
#endif