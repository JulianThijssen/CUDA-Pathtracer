#pragma once

#include "Renderer.h"
#include "Vector3f.h"
#include "Camera.h"

#include "kernel.cuh"

class CudaRenderer : protected Renderer
{
public:
    CudaRenderer();
    virtual void init(const Scene& scene) override;
    virtual void resize(Size size) override;
    virtual void update() override;
    virtual void destroy() override;

    Vector3f* final;

private:
    GPU_Scene gpu_scene;
    
    Camera camera;

    Vector3f* out;
    Vector3f* accumulation;
    
    Vector3f* dev_out;

    curandState *d_state;

    int iterations = 0;
};
