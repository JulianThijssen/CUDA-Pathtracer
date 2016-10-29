#include "kernel.cuh"
#include "curand.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <string>

#include "Ray.h"

#define PI 3.14159265
#define ONE_OVER_PI 0.318309886

#define CUDA __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define CAMERA_FAR 10000
#define ITERATIONS 5
#define NUM_THREADS 64
#define EPSILON 0.001
#define ABSORPTION 0.25

CUDA struct Basis {
	Vector3f x;
	Vector3f y;
	Vector3f z;
};

float randf() {
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

__device__ Vector3f uniformHemisphereSample(unsigned int idx, curandState *state, Vector3f n) {
	float x = curand_normal(&state[idx]) * 2 - 1;
	float y = curand_normal(&state[idx]) * 2 - 1;
	float z = curand_normal(&state[idx]) * 2 - 1;

	Vector3f rand(x, y, z);
	rand.normalise();

	return (dot(rand, n) > 0) ? rand: -rand;
}

__device__ Vector3f cosineHemisphereSample(unsigned int idx, curandState *state, Vector3f n) {
	float u1 = curand_uniform(&state[idx]);
	float u2 = curand_uniform(&state[idx]);

	float phi = 2 * PI * u2;
	float cosTheta = sqrtf(1.0 - u1);
	float sinTheta = sqrtf(1.0 - cosTheta * cosTheta);

	float x = cosf(phi) * sinTheta;
	float y = sinf(phi) * sinTheta;
	float z = cosTheta;

	Vector3f h(n.x, n.y, n.z);
	Vector3f t = h;
	if (fabsf(t.x) <= fabsf(t.y) && fabsf(t.x) <= fabsf(t.z))
		t.x = 1.0;
	else if (fabsf(t.y) <= fabsf(t.x) && fabsf(t.y) <= fabsf(t.z))
		t.y = 1.0;
	else
		t.z = 1.0;

	Vector3f b = cross(n, t);
	b.normalise();
	t = cross(b, n);

	float nx = t.x * x + b.x * y + n.x * z;
	float ny = t.y * x + b.y * y + n.y * z;
	float nz = t.z * x + b.z * y + n.z * z;

	return Vector3f(nx, ny, nz);
}

__global__ void setup_kernel(curandState *state) {
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(0, idx, 0, &state[idx]);
}

__device__ void directLighting(const unsigned int idx, const Ray &ray,
	Vector3f &rad, const Scene &scene, curandState *state)
{
	Vector3f reflRad = Vector3f(1, 1, 1);
	
	Ray r(ray.o, ray.d);

	// Scene intersection
	float t;
	Vector3f n;
	Mesh *mesh;
	bool hit;

	hit = scene.intersect(r, &mesh, t, n);

	// If no hit was found, there will be no lighting
	if (!hit)
		return;

	Material mat = scene.dev_materials[mesh->materialIndex];
	if (mat.emission.length() > EPSILON) {
		rad += reflRad * mat.emission;
		return;
	}

	// Find the light
	Mesh *light = 0;
	for (unsigned int i = 0; i < scene.meshCount; i++) {
		Mesh m = scene.dev_meshes[i];
		Material lightMat = scene.dev_materials[m.materialIndex];
		if (lightMat.emission.length() > 1) {
			light = &scene.dev_meshes[i];
			break;
		}
	}

	// Get a random sample on the light
	Vector3f sample = light->getRandomSample(idx, state);

	// create a shadow ray to the light sample
	r.o = r.o + r.d * t;
	r.d = (sample - r.o).normalise();

	// Apply BRDF
	float cos = dot(n, r.d);
	float brdf = 2.0f;
	reflRad *= mat.albedo * cos * brdf;
	
	// Scene intersection
	hit = scene.intersect(r, &mesh, t, n);
	
	// If no hit was found, there will be no lighting
	if (!hit) {
		return;
	}
	
	mat = scene.dev_materials[mesh->materialIndex];

	r.o = r.o + r.d * t;
	float diff = (r.o - sample).length();

	float G = (cos * dot(n, -r.d)) / (t * t);

	// Check if we hit the light
	if (mat.emission.length() > EPSILON) {
		rad += reflRad * mat.emission * G / (1.0f / 13560);
		return;
	}
}

__device__ void indirectLighting(const unsigned int idx, const Ray &ray,
	Vector3f &rad, const Scene &scene, curandState *state)
{
	Ray r(ray.o, ray.d);

	Vector3f reflRad = Vector3f(1, 1, 1);
	while (true) {
		float p = curand_uniform(&state[idx]);

		// Russian Roulette
		if (p < ABSORPTION) {
			return;
		}

		// Scene intersection
		float t;
		Vector3f n;
		Mesh *mesh;

		scene.intersect(r, &mesh, t, n);
		Material mat = scene.dev_materials[mesh->materialIndex];

		if (t < CAMERA_FAR) {
			if (mat.emission.length() > EPSILON) {
				// We hit a light, set the total radiance
				float rrWeight = 1 / (1 - ABSORPTION);
				rad += reflRad * mat.emission * rrWeight * 2.0 * PI;
				return;
			}

			// Generate new ray from intersection
			r.o = r.o + r.d * t;
			r.d = cosineHemisphereSample(idx, state, n);

			reflRad *= mat.albedo;
		}
		else {
			// The ray escaped, no contribution
			return;
		}
	}
}

__global__ void traceKernel(float* out, const int w, const int h,
	const Vector3f o, const Basis basis, const Scene scene, curandState *state)
{
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int x = idx % w;
	unsigned int y = idx / w;

	float uvx = 2 * ((float)x / w) - 1;
	float uvy = 2 * ((float)y / h) - 1;

	float aspect = (float)w / h;

	Vector3f rayO = o + (basis.x * uvx) + (basis.y * uvy);
	Vector3f rayD = ((((basis.x * aspect * uvx) + (basis.y * uvy)) * 0.33135) + basis.z).normalise();
	Ray ray(rayO, rayD);

	Vector3f rad(0, 0, 0);

	directLighting(idx, ray, rad, scene, state);
	indirectLighting(idx, ray, rad, scene, state);

	out[y * w * 3 + x * 3 + 0] += rad.x;
	out[y * w * 3 + x * 3 + 1] += rad.y;
	out[y * w * 3 + x * 3 + 2] += rad.z;
}

cudaError_t uploadMesh(Scene &scene)
{
	Mesh* mesh = new Mesh[scene.meshCount];
	for (unsigned int i = 0; i < scene.meshCount; i++) {
		memcpy(&mesh[i], &scene.getMesh(i), sizeof(Mesh));
	}
	
	cudaError_t cudaStatus;

	Mesh* h_mesh = new Mesh[scene.meshCount];
	
	for (unsigned int i = 0; i < scene.meshCount; i++) {
		printf("Number of vertices: %d, Number of normals: %d, Number of faces: %d\n", mesh[i].numVerts, mesh[i].numNorms, mesh[i].numFaces);
		
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

		h_mesh[i].materialIndex = mesh[i].materialIndex;
		h_mesh[i].vertices = vertices;
		h_mesh[i].normals = normals;
		h_mesh[i].faces = faces;
		h_mesh[i].numVerts = mesh[i].numVerts;
		h_mesh[i].numNorms = mesh[i].numNorms;
		h_mesh[i].numFaces = mesh[i].numFaces;
	}

	cudaStatus = cudaMalloc((void**)&scene.dev_meshes, scene.meshCount * sizeof(Mesh));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Meshes cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(scene.dev_meshes, h_mesh, scene.meshCount * sizeof(Mesh), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Meshes cudaMemcpy failed!");
	}
	cudaStatus = cudaMalloc((void**)&scene.dev_materials, scene.materialCount * sizeof(Material));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Materials cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(scene.dev_materials, &scene.materials[0], scene.materialCount * sizeof(Material), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Materials cudaMemcpy failed!");
	}

	return cudaStatus;
}

cudaError_t init(uint w, uint h, curandState** d_state) {
	cudaError_t cudaStatus;

	unsigned int blockSize = NUM_THREADS;
	unsigned int gridSize = (w * h) / NUM_THREADS + ((w * h) % NUM_THREADS == 0 ? 0 : 1);

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

cudaError_t trace(float** dev_out, const Vector3f& o, const Vector3f& d, uint width, uint height, const Scene &scene, curandState* d_state) {
	cudaError_t cudaStatus;
	
	unsigned int blockSize = NUM_THREADS;
	unsigned int gridSize = (width * height) / NUM_THREADS + ((width * height) % NUM_THREADS == 0 ? 0 : 1);

	Vector3f cz = d;
	Vector3f cy(0, 1, 0);
	Vector3f cx = cross(cz, cy).normalise();
	cy = cross(cx, cz);
	Basis basis = { cx, cy, cz };

	// Launch a kernel on the GPU with one thread for each element.
	traceKernel << <gridSize, blockSize >> >(*dev_out, width, height, o, basis, scene, d_state);
	
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

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "Freeing CUDA failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	return cudaStatus;
}
