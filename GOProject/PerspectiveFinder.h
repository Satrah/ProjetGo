#pragma once

#include "Image.h"

namespace GOProject
{
	class PerspectiveFinder : public Image<uchar>
	{
	public:
		PerspectiveFinder(int memoryPoints = 20) : _memoryPoints(memoryPoints) {}
		bool HomographyTransform();
		int _memoryPoints;
		std::vector<cv::Point2f> _pointsOrig;
		std::vector<cv::Point2f> _pointsDest;
	};
}