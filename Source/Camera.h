#pragma once

#include "cuda_runtime.h"

#define CUDA __host__ __device__

#include "Vector3f.h"

CUDA class Camera
{
public:
    Camera(Vector3f position, Vector3f direction);

    Vector3f position;
    Vector3f direction;
};
