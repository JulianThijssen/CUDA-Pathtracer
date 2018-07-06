#pragma once
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include "Scene.h"
#include "Mesh.h"
#include "Size.h"
#include "Vector3f.h"

class Camera;

typedef unsigned int uint;

//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {\
    cudaError_t e = cudaGetLastError();\
    if (e != cudaSuccess) {\
        printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));\
        exit(0);\
    }\
}

void kernelInit(Size size, curandState** d_state);
bool uploadMesh(const Scene& scene, GPU_Scene& gpu_scene);
cudaError_t kernelDestroy(curandState** d_state);
cudaError_t trace(Vector3f** dev_out, const Camera& camera, Size size, const GPU_Scene &scene, curandState* d_state);

#endif /* KERNEL_CUH */
