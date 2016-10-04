#include "kernel.cuh"
#include "curand.h"

#include "ModelLoader.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <string>

#define PI 3.14159265
#define ONE_OVER_PI 0.318309886

#define CUDA __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define CAMERA_FAR 10000
#define NUM_MESHES 8
#define ITERATIONS 5
#define NUM_THREADS 32
#define EPSILON 0.001
#define ABSORPTION 0.25

CUDA struct Ray {
	Vector3f o;
	Vector3f d;
};

float randf() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

__device__ Vector3f generateVector(unsigned int idx, curandState *state, Vector3f n) {
	float x = curand_normal(&state[idx]) * 2 - 1;
	float y = curand_normal(&state[idx]) * 2 - 1;
	float z = curand_normal(&state[idx]) * 2 - 1;

	Vector3f rand(x, y, z);
	rand.normalise();

	return (dot(rand, n) > 0) ? rand: -rand;
}

__global__ void setup_kernel(curandState *state) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

__device__ void indirectLighting(const unsigned int idx, const Ray &ray,
	Vector3f &rad, Mesh* meshes, curandState *state)
{
	Vector3f o = ray.o;
	Vector3f d = ray.d;

	Vector3f reflRad = Vector3f(1, 1, 1);
	while (true) {
		float p = curand_uniform(&state[idx]);

		// Russian Roulette
		if (p < ABSORPTION) {
			return;
		}

		// Scene intersection
		float closest_t = CAMERA_FAR;
		Vector3f hit_n;
		Mesh* mesh;

		for (int j = 0; j < NUM_MESHES; j++) {
			Vector3f n(0, 0, 0);

			float t = intersect(o + d*EPSILON, d, meshes[j], n);
			if (t > 0 && t < closest_t) {
				closest_t = t;
				hit_n.set(n.x, n.y, n.z);
				mesh = &meshes[j];
			}
		}

		if (closest_t < CAMERA_FAR) {
			if (mesh->emission > EPSILON) {
				// We hit a light, set the total radiance
				float rrWeight = 1 / (1 - ABSORPTION);
				rad = reflRad * mesh->emission * rrWeight;
				return;
			}

			// Generate new ray from intersection
			o = o + d * closest_t;
			d = generateVector(idx, state, hit_n);

			float cos = dot(hit_n, d);
			float brdf = 2.0f;// ONE_OVER_PI;
			reflRad *= mesh->albedo * cos * brdf;
		}
		else {
			// The ray escaped, no contribution
			return;
		}
	}
}

__global__ void traceKernel(float* out, const int w, const int h,
	const Vector3f o, const Vector3f cx, const Vector3f cy, const Vector3f cz,
	Mesh* meshes, curandState *state)
{
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int x = idx % w;
	unsigned int y = idx / w;

	float uvx = 2 * ((float)x / w) - 1;
	float uvy = 2 * ((float)y / h) - 1;

	float aspect = (float)w / h;

	Vector3f rayO = o + (cx * uvx) + (cy * uvy);
	Vector3f rayD = ((((cx * aspect * uvx) + (cy * uvy)) * 0.33135) + cz).normalise();
	Ray ray = { rayO, rayD };

	Vector3f rad(0, 0, 0);

	indirectLighting(idx, ray, rad, meshes, state);

	out[y * w * 3 + x * 3 + 0] += rad.x;
	out[y * w * 3 + x * 3 + 1] += rad.y;
	out[y * w * 3 + x * 3 + 2] += rad.z;
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
