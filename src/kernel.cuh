#pragma once
#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>

#include "Mesh.h"
#include "Vector3f.h"

typedef unsigned int uint;

cudaError_t init(uint w, uint h, curandState** d_state);
cudaError_t uploadMesh(Mesh** meshes, unsigned int &meshCount);
cudaError_t destroy(curandState** d_state);
cudaError_t trace(float** dev_out, const Vector3f& dev_o, const Vector3f& dev_d, uint w, uint h, Mesh* meshes, const unsigned int meshCount, curandState* d_state);

#endif /* KERNEL_CUH */
