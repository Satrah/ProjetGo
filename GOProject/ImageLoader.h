#ifndef _HEADER_IMAGE_LOADER
#define _HEADER_IMAGE_LOADER

namespace GOProject
{

class ImageLoader
{
public:
    ImageLoader() {}
    bool Load(const char* imageFile);
    void DebugDisplay();
};

};

#endif