#pragma once

class Size
{
public:
    Size()
        : width(1), height(1)
    { }

    Size(unsigned int width, unsigned int height)
        : width(width), height(height)
    { }

    unsigned int width, height;
};
