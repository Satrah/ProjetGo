#pragma once

#include "Image.h"

namespace GOProject
{
	class PerspectiveFinder : public Image<uchar>
	{
	public:
		PerspectiveFinder(int cornersOffset, int memoryPoints = 20) : _cornersOffset(cornersOffset), _memoryPoints(memoryPoints) {}
		bool HomographyCalibrate();
		void HomographyTransform();
		cv::Mat GetHomography() const { return _homography; }
		int GetCornersOffset() const { return _cornersOffset; }
	protected:
		int _memoryPoints;
		std::vector<cv::Point2f> _pointsOrig;
		std::vector<cv::Point2f> _pointsDest;
		cv::Mat _homography;
		int _cornersOffset;
	};
}