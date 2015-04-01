#pragma once

#include "Image.h"

namespace GOProject
{
	class PerspectiveFinder : public Image<uchar>
	{
	public:
		PerspectiveFinder() {}
		bool HomographyTransform();
	};
}