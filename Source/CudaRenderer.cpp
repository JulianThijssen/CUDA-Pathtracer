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

void CudaRenderer::resize(Size size)
{
    windowSize = size;

    unsigned int numPixels = windowSize.width * windowSize.height;
    out = new Vector3f[numPixels];
    accumulation = new Vector3f[numPixels];
    final = new Vector3f[numPixels];
    dev_out = 0;

    // Make output
    cudaMalloc((void**)&dev_out, numPixels * sizeof(Vector3f));
    cudaCheckError();

    // Init the kernel
    kernelInit(windowSize, &d_state);
}

void CudaRenderer::update()
{
    uint numPixels = windowSize.width * windowSize.height;

    // Add vectors in parallel.
    trace(&dev_out, camera, windowSize, gpu_scene, d_state);
    cudaCheckError();

    iterations++;

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(out, dev_out, numPixels * sizeof(Vector3f), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Add the output to the accumulation buffer
    for (unsigned int i = 0; i < numPixels; i++) {
        accumulation[i] += out[i];
    }

    // Divide the accumulated buffer by the iterations
    for (unsigned int i = 0; i < numPixels; i++) {
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
