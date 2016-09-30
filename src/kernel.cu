#include "kernel.cuh"
#include "curand.h"

#include "ModelLoader.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <string>

#define HOST __host__
#define DEVICE __device__
#define CAMERA_FAR 10000
#define NUM_MESHES 8
#define ITERATIONS 5
#define NUM_THREADS 32
#define EPSILON 0.0001

float randf() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

__device__ Vector3f generateVector(curandState *state, unsigned int idx, Vector3f n) {
	float x = curand_uniform(&state[idx]) * 2 - 1;
	float y = curand_uniform(&state[idx]) * 2 - 1;
	float z = curand_uniform(&state[idx]) * 2 - 1;

	Vector3f rand(x, y, z);
	rand.normalise();

	return (dot(rand, n) > 0) ? rand: -rand;
}

__global__ void setup_kernel(curandState *state) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

__global__ void traceKernel(float* out, const int w, const int h,
	const Vector3f o, const Vector3f cx, const Vector3f cy, const Vector3f cz,
	Mesh* meshes, curandState *state)
{
	//int i = threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int x = idx % w;
	unsigned int y = idx / w;

	float uvx = 2 * ((float)x / w) - 1;
	float uvy = 2 * ((float)y / h) - 1;

	float aspect = (float) w / h;

	Vector3f rayO = o + (cx * uvx) + (cy * uvy);
	Vector3f rayD = ((((cx * aspect * uvx) + (cy * uvy)) * 0.33135) + cz).normalise();

	Vector3f rad(1, 1, 1);

	for (int k = 0; k < ITERATIONS; k++) {
		// Scene intersection
		float min_t = CAMERA_FAR;
		Vector3f min_n(0, 0, 0);
		Mesh* mesh;

		for (int j = 0; j < NUM_MESHES; j++) {
			Vector3f n(0, 0, 0);

			float t = intersect(rayO+rayD*EPSILON, rayD, meshes[j], n);
			if (t > 0 && t < min_t) {
				min_t = t;
				min_n.set(n.x, n.y, n.z);
				mesh = &meshes[j];
			}
		}

		if (min_t < CAMERA_FAR) {
			if (mesh->emission > 0.5) {
				rad *= mesh->emission;
				break;
			}
			rayO = rayO + rayD * min_t; // Intersection point
			rayD = generateVector(state, idx, min_n);

			float cos = dot(min_n, rayD);
			Vector3f brdf = mesh->albedo * (2 * cos);
			rad *= brdf;
		}
		else {
			rad.set(0, 0, 0);
			break;
		}
		if (k == ITERATIONS - 1) {
			rad.set(0, 0, 0);
			break;
		}
	}
	out[y * w * 3 + x * 3 + 0] += rad.x;
	out[y * w * 3 + x * 3 + 1] += rad.y;
	out[y * w * 3 + x * 3 + 2] += rad.z;
}

__global__ void accumKernel(float* out, const int w, const int h, float* in, const int k) {
	
}

cudaError_t uploadMesh(Mesh** meshes)
{
	Mesh* mesh = loadMesh(std::string("path"));
	
	cudaError_t cudaStatus;

	Mesh* h_mesh = new Mesh[8];
	
	for (int i = 0; i < 8; i++) {
		printf("Sizes: %d, %d, %d, %d, %f\n", mesh[i].numVerts, mesh[i].numNorms, mesh[i].numFaces, i, mesh[i].emission);
		printf("%s, %s\n", mesh[i].vertices[0].str().c_str(), mesh[i].normals[0].str().c_str());
		
		Vector3f* vertices = 0;
		cudaStatus = cudaMalloc((void**)&vertices, mesh[i].numVerts * sizeof(Vector3f));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Vertices cudaMalloc failed!");
		}
		cudaStatus = cudaMemcpy(vertices, mesh[i].vertices, mesh[i].numVerts * sizeof(Vector3f), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Vertices cudaMemcpy failed!");
		}

		Vector3f* normals = 0;
		cudaStatus = cudaMalloc((void**)&normals, mesh[i].numNorms * sizeof(Vector3f));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Normals cudaMalloc failed!");
		}
		cudaStatus = cudaMemcpy(normals, mesh[i].normals, mesh[i].numNorms * sizeof(Vector3f), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Vertices cudaMemcpy failed!");
		}

		Face* faces = 0;
		cudaStatus = cudaMalloc((void**)&faces, mesh[i].numFaces * sizeof(Face));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Faces cudaMalloc failed!");
		}
		cudaStatus = cudaMemcpy(faces, mesh[i].faces, mesh[i].numFaces * sizeof(Face), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "Faces cudaMemcpy failed!");
		}

		h_mesh[i].vertices = vertices;
		h_mesh[i].normals = normals;
		h_mesh[i].faces = faces;
		h_mesh[i].numVerts = mesh[i].numVerts;
		h_mesh[i].numNorms = mesh[i].numNorms;
		h_mesh[i].numFaces = mesh[i].numFaces;
		h_mesh[i].emission = mesh[i].emission;
		h_mesh[i].albedo = mesh[i].albedo;
	}

	cudaStatus = cudaMalloc((void**)meshes, 8 * sizeof(Mesh));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Meshes cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(*meshes, h_mesh, 8 * sizeof(Mesh), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Meshes cudaMemcpy failed!");
	}

	return cudaStatus;
}

cudaError_t init(curandState** d_state) {
	cudaError_t cudaStatus;

	unsigned int blockSize = NUM_THREADS;
	unsigned int gridSize = (512 * 512) / NUM_THREADS + ((512 * 512) % NUM_THREADS == 0 ? 0 : 1);

	cudaMalloc(d_state, gridSize * blockSize * sizeof(curandState));

	setup_kernel << <gridSize, blockSize >> >(*d_state);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "traceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	return cudaStatus;
}

cudaError_t trace(float** dev_out, const Vector3f& o, const Vector3f& d, uint width, uint height, Mesh* meshes, curandState* d_state) {
	cudaError_t cudaStatus;
	
	unsigned int blockSize = NUM_THREADS;
	unsigned int gridSize = (width * height) / NUM_THREADS + ((width * height) % NUM_THREADS == 0 ? 0 : 1);

	Vector3f cz = d;
	Vector3f cy(0, 1, 0);
	Vector3f cx = cross(cz, cy).normalise();
	cy = cross(cx, cz);

	// Launch a kernel on the GPU with one thread for each element.
	traceKernel << <gridSize, blockSize >> >(*dev_out, width, height, o, cx, cy, cz, meshes, d_state);
	
	//accumKernel << <gridSize, blockSize >> >(*dev_out, 512, 512, dev_out, 1);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "traceKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

Error:
	return cudaStatus;
}

cudaError_t destroy(curandState** d_state) {
	cudaError_t cudaStatus;

	cudaFree(d_state);

	return cudaStatus;
}
