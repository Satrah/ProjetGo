#pragma once

#include "Image.h"

namespace GOProject
{
	class PerspectiveFinder : public Image<uchar>
	{
	public:
		PerspectiveFinder(Image<uchar> img) : Image<uchar>(img) {}
		bool HomographyTransform();
	};
}