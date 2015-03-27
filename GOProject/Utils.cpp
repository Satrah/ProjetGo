#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/internal.hpp>

void cvEqualizeHist(const CvArr* srcarr, CvArr* dstarr, CvMat* mask)
{
	using namespace cv;

	CvMat sstub, *src = cvGetMat(srcarr, &sstub);
	CvMat dstub, *dst = cvGetMat(dstarr, &dstub);

	CV_Assert(CV_ARE_SIZES_EQ(src, dst) && CV_ARE_TYPES_EQ(src, dst) &&
		CV_MAT_TYPE(src->type) == CV_8UC1);

	CV_Assert(CV_ARE_SIZES_EQ(src, mask) && CV_MAT_TYPE(mask->type) == CV_8UC1);

	CvSize size = cvGetMatSize(src);
	if (CV_IS_MAT_CONT(src->type & dst->type))
	{
		size.width *= size.height;
		size.height = 1;
	}
	int x, y;
	const int hist_sz = 256;
	int hist[hist_sz];
	memset(hist, 0, sizeof(hist));

	for (y = 0; y < size.height; y++)
	{
		const uchar* sptr = src->data.ptr + src->step*y;
		const uchar* mptr = mask->data.ptr + mask->step*y;
		for (x = 0; x < size.width; x++)
		if (mptr[x]) hist[sptr[x]]++;
	}

	float scale = 255.f / (cvCountNonZero(mask));
	int sum = 0;
	uchar lut[hist_sz + 1];

	for (int i = 0; i < hist_sz; i++)
	{
		sum += hist[i];
		int val = cvRound(sum*scale);
		lut[i] = CV_CAST_8U(val);
	}

	lut[0] = 0;
	cvSetZero(dst);
	for (y = 0; y < size.height; y++)
	{
		const uchar* sptr = src->data.ptr + src->step*y;
		const uchar* mptr = mask->data.ptr + mask->step*y;
		uchar* dptr = dst->data.ptr + dst->step*y;
		for (x = 0; x < size.width; x++)
		if (mptr[x]) dptr[x] = lut[sptr[x]];
	}
}

void equalizeHist(cv::InputArray _src, cv::OutputArray _dst, cv::InputArray _mask)
{
	using namespace cv;

	Mat src = _src.getMat().clone();
	_dst.create(src.size(), src.type());
	Mat dst = _dst.getMat();
	Mat mask;
	if (_mask.empty()) mask = Mat::ones(src.size(), CV_8UC1);
	else mask = _mask.getMat();
	CvMat _csrc = src, _cdst = dst, _cmask = mask;
	cvEqualizeHist(&_csrc, &_cdst, &_cmask);
}

// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2,
	cv::Point2f &r)
{
	cv::Point2f x = o2 - o1;
	cv::Point2f d1 = p1 - o1;
	cv::Point2f d2 = p2 - o2;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (abs(cross) < /*EPS*/1e-8)
		return false;

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}


void fillRectangle(cv::OutputArray mat, cv::RotatedRect rect)
{
	cv::Scalar color(1);
	rect.size.height += 100;
	rect.size.width += 100;
	cv::Point2f rectPoints[4];
	cv::Point polyPoints[4];
	int nPoints[] = { 4 };
	rect.points(rectPoints);
	for (int i = 0; i < 4; ++i)
		polyPoints[i] = cv::Point(rectPoints[i].x, rectPoints[i].y);
	const cv::Point* ppt[1] = { polyPoints };
	fillPoly(mat.getMat(), (const cv::Point**)ppt, nPoints, 1, color);
}