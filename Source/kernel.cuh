#pragma once
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include "Scene.h"
#include "Mesh.h"
#include "Vector3f.h"

typedef unsigned int uint;

cudaError_t init(uint w, uint h, curandState** d_state);
cudaError_t uploadMesh(Scene &scene, GPU_Scene& gpu_scene);
cudaError_t destroy(curandState** d_state);
cudaError_t trace(Vector3f** dev_out, const Vector3f& dev_o, const Vector3f& dev_d, uint w, uint h, const GPU_Scene &scene, curandState* d_state);

#endif /* KERNEL_CUH */
