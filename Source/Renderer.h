#pragma once

#include "Scene.h"

class Renderer
{
    virtual void init(const Scene& scene) = 0;
    virtual void resize(unsigned int width, unsigned int height) = 0;
    virtual void update() = 0;
    virtual void destroy() = 0;
};
