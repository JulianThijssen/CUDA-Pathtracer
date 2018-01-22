#include "Window.h"

#include "Scene.h"
#include "ModelLoader.h"

#include <glad/glad.h>
#include "kernel.cuh"
#include "curand.h"

#include <ctime>

int main()
{
	Window window("Path Tracer", 512, 512);
	Scene scene;
    GPU_Scene gpu_scene;
	
	Vector3f o(278, 273, -800);
	const Vector3f d(0, 0, 1);


	unsigned int width, height;
	window.getSize(width, height);

	Vector3f* out = new Vector3f[width * height];
	Vector3f* dev_out = 0;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);

	// Upload the scene
	loadScene(scene, std::string("res/cornell_box.obj"));

	cudaStatus = uploadMesh(scene, gpu_scene);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "uploadtriangle failed!");
		return 1;
	}

	// Make output
	cudaStatus = cudaMalloc((void**)&dev_out, width * height * sizeof(Vector3f));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_out cudaMalloc failed!");
		exit(1);
	}

	// Init the kernel
	curandState *d_state;
	init(width, height, &d_state);

	clock_t begin = clock();
	int frames = 0;
	int iterations = 0;

	while (!window.isClosed()) {
		clock_t end = clock();
		double elapsed = double(end - begin) / CLOCKS_PER_SEC;
		if (elapsed >= 1) {
			begin = clock();
			printf("Frames: %d\n", frames);
			frames = 0;
		}
		
		// Add vectors in parallel.
		cudaStatus = trace(&dev_out, o, d, width, height, gpu_scene, d_state);
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
		
		// Divide the accumulated buffer by the iterations
		for (unsigned int i = 0; i < width * height; i++) {
			out[i] /= iterations;
		}

		if (iterations % 1000 == 0) {
			printf("%d iterations\n", iterations);
		}

		glDrawPixels(width, height, GL_RGB, GL_FLOAT, out);

		window.update();
		frames++;
	}

	delete out;

	cudaStatus = cudaFree(dev_out);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Failed to free device pixelbuffer!");
		exit(1);
	}
	destroy(&d_state);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
