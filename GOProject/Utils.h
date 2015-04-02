#pragma once

#include <opencv2/imgproc/imgproc.hpp>


namespace GOProject
{
	typedef std::vector<cv::Vec4i> LinesVec;
};

void cvEqualizeHist(const CvArr* srcarr, CvArr* dstarr, CvMat* mask);
void equalizeHist(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask);
// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r);
void fillRectangle(cv::OutputArray mat, cv::RotatedRect rect);
std::string type2str(int type);