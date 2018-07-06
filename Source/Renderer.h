#pragma once

#include "Scene.h"

#include "Size.h"

class Renderer
{
public:
    virtual void init(const Scene& scene) = 0;
    virtual void resize(Size size) = 0;
    virtual void update() = 0;
    virtual void destroy() = 0;

protected:
    Size windowSize;
};
