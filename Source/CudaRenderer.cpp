#include "CudaRenderer.h"

#include <iostream>

CudaRenderer::CudaRenderer()
    : camera(Vector3f(), Vector3f())
{

}

void CudaRenderer::init(const Scene& scene)
{
    std::cout << "Uploading scene.." << std::endl;
    bool success = uploadMesh(scene, gpu_scene);

    if (!success) {
        fprintf(stderr, "uploadtriangle failed!");
        //return 1;
    }

    camera = Camera(Vector3f(278, 273, -600), Vector3f(0, 0, 1));

    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
}

void CudaRenderer::resize(unsigned int width, unsigned int height)
{
    this->width = width;
    this->height = height;
    out = new Vector3f[width * height];
    accumulation = new Vector3f[width * height];
    final = new Vector3f[width * height];
    dev_out = 0;

    // Make output
    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((void**)&dev_out, width * height * sizeof(Vector3f));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_out cudaMalloc failed!");
        exit(1);
    }

    // Init the kernel
    kernelInit(width, height, &d_state);
}

void CudaRenderer::update()
{
    cudaError_t cudaStatus;

    // Add vectors in parallel.
    cudaStatus = trace(&dev_out, camera, width, height, gpu_scene, d_state);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "trace failed!");
        exit(1);
    }

    iterations++;

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, width * height * sizeof(Vector3f), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "dev_out -> out cudaMemcpy failed!");
        exit(1);
    }

    // Add the output to the accumulation buffer
    for (unsigned int i = 0; i < width * height; i++) {
        accumulation[i] += out[i];
    }

    // Divide the accumulated buffer by the iterations
    for (unsigned int i = 0; i < width * height; i++) {
        final[i] = accumulation[i] / iterations;
    }

    if (iterations % 1000 == 0) {
        printf("%d iterations\n", iterations);
    }
}

void CudaRenderer::destroy()
{
    delete out;

    cudaError_t cudaStatus;
    cudaStatus = cudaFree(dev_out);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to free device pixelbuffer!");
        exit(1);
    }
    kernelDestroy(&d_state);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        //return 1;
    }
}
