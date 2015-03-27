#pragma once

#include <opencv2/imgproc/imgproc.hpp>

void cvEqualizeHist(const CvArr* srcarr, CvArr* dstarr, CvMat* mask);
void equalizeHist(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask = cv::noArray());