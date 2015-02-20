#include <iostream>
#include "GOProject/ImageLoader.h"

int main()
{
    GOProject::ImageLoader loader;
    loader.Load("img/test1.jpg");
    loader.DebugDisplay();
    return 0;
}