#ifndef _HEADER_OPENCV_IMAGE_WRAPPER
#define _HEADER_OPENCV_IMAGE_WRAPPER

#include <vector>
#include <opencv2/imgproc/imgproc.hpp>

namespace GOProject
{
	template <typename T>
	class Image : public cv::Mat {
	public:
		// Constructors
		Image() {}
		Image(const cv::Mat& A):cv::Mat(A) {}
		Image(int w,int h,int type):cv::Mat(h,w,type) {}
		// Accessors
		inline T operator()(int x,int y) const { return at<T>(y,x); }
		inline T& operator()(int x,int y) { return at<T>(y,x); }
		inline T operator()(const cv::Point& p) const { return at<T>(p.y,p.x); }
		inline T& operator()(const cv::Point& p) { return at<T>(p.y,p.x); }
		//
		inline int width() const { return cols; }
		inline int height() const { return rows; }
		// To display a floating type image
		Image<uchar> greyImage() const {
			double minVal, maxVal;
			minMaxLoc(*this,&minVal,&maxVal);
			Image<uchar> g;
			convertTo(g, CV_8U, 255.0/(maxVal - minVal), -minVal * 255.0/(maxVal - minVal));
			return g;
		}
	};
};
#endif