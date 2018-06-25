#include "kernel.cuh"
#include "curand.h"

#include <stdio.h>
#include <cstdlib>
#include <ctime>
#include <cfloat>
#include <string>

#include "Camera.h"
#include "Ray.h"
#include "BRDF.h"

#define CUDA __host__ __device__
#define HOST __host__
#define DEVICE __device__
#define NUM_THREADS 64
#define ABSORPTION 0.25

__device__ HitInfo trace(const GPU_Scene& scene, Ray ray);
__device__ Vector3f computeRadiance(const GPU_Scene& scene, Ray r, const Vector3f& camPos, const unsigned int idx, curandState *state);

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

    return (dot(rand, n) > 0) ? rand : -rand;
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


__device__ Vector3f directIllumination(const GPU_Scene& scene, Vector3f x, HitInfo info, const unsigned int idx, curandState *state) {
    Vector3f Radiance;

    Material mat = scene.dev_materials[info.mesh->materialIndex];

    // Find the light
    Mesh *light = 0;
    for (unsigned int i = 0; i < scene.meshCount; i++) {
        Mesh& m = scene.dev_meshes[i];
        Material lightMat = scene.dev_materials[m.materialIndex];
        if (lightMat.emission.length() > 1) {
            light = &scene.dev_meshes[i];
            break;
        }
    }

    // Get a random sample on the light
    Vector3f sample = light->getRandomSample(idx, state);

    // create a shadow ray to the light sample
    Ray r(x, (sample - x).normalise());

    Vector3f L = r.d;
    //Vector3f H = (L + V).normalise();

    float cos = CosTheta(info.n, L);

    // Apply BRDF
    Vector3f brdf = BRDF(info.n, L, mat);

    // Scene intersection
    info = trace(scene, r);

    // If no hit was found, there will be no lighting
    if (!info.hit) {
        return Radiance;
    }

    mat = scene.dev_materials[info.mesh->materialIndex];

    x = r.o + r.d * info.t;

    float G = (cos * CosTheta(info.n, -L)) / (info.t * info.t);

    // Check if we hit the light
    if (mat.emission.length() > EPSILON) {
        Radiance = (mat.emission * brdf * G * 13650);
    }
    return Radiance;
}

// Russian Roulette
__device__ bool isAbsorbed(curandState* state) {
    float p = curand_uniform(state);

    return p < ABSORPTION;
}

__device__ HitInfo trace(const GPU_Scene& scene, Ray ray) {
    return scene.intersect(ray);
}

__device__ Vector3f computeRadiance(const GPU_Scene& scene, Ray r, const Camera& camera, const unsigned int idx, curandState *state) {
    Vector3f Radiance;

    Vector3f PreRadiance[30];
    HitInfo hits[30];
    Vector3f psi[30];
    int index = 0;

    float rrWeight = 1.0f / (1.0f - ABSORPTION);
    do {
        HitInfo info = trace(scene, r);
        hits[index] = info;

        if (info.hit && info.t < camera.zFar) {
            Material mat = scene.dev_materials[info.mesh->materialIndex];

            Vector3f x = r.o + r.d * info.t;

            Vector3f Ld;
            Ld += mat.emission;
            Ld += directIllumination(scene, x, info, idx, state);

            PreRadiance[index] = Ld;

            Ray newRay(x, cosineHemisphereSample(idx, state, info.n));
            r.o = newRay.o;
            r.d = newRay.d;
            psi[index] = r.d;

            if (isAbsorbed(&state[idx])) break;

            index++;
        }
        else {
            PreRadiance[index] = Vector3f(0);
            break;
        }
    } while (index < 30);

    Radiance = PreRadiance[index];
    if (index > 0) {
        for (int i = index - 1; i >= 0; i--) {
            HitInfo info = hits[i];
            Vector3f Ld = PreRadiance[i];
            Vector3f L = psi[i];
            //Vector3f V = (camPos - x).normalise();

            Material mat = scene.dev_materials[info.mesh->materialIndex];

            // Apply BRDF
            Vector3f brdf = BRDF(info.n, L, mat);

            float cos = CosTheta(info.n, L);

            Radiance = Ld + (Radiance * brdf * cos * rrWeight);
        }
    }

    return Radiance;
}

__global__ void traceKernel(Vector3f* out, const int w, const int h,
    const Camera camera, const Basis basis, const GPU_Scene scene, curandState *state)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int x = idx % w;
    unsigned int y = idx / w;

    float uvx = 2 * ((float)x / w) - 1;
    float uvy = 2 * ((float)y / h) - 1;

    float aspect = (float)w / h;

    Vector3f rayO = camera.position + (basis.x * uvx) + (basis.y * uvy);
    Vector3f rayD = ((((basis.x * aspect * uvx) + (basis.y * uvy)) * 0.33135) + basis.z).normalise();
    Ray ray(rayO, rayD);

    Vector3f Radiance = computeRadiance(scene, ray, camera, idx, state);

    //Radiance *= Vector3f(2.0f) / ((Radiance / 2.0f) + 1);

    out[idx] += Radiance;
}

cudaError_t uploadMesh(Scene &scene, GPU_Scene& gpu_scene)
{
    Mesh* mesh = new Mesh[scene.meshCount];
    for (unsigned int i = 0; i < scene.meshCount; i++) {
        memcpy(&mesh[i], scene.getMesh(i), sizeof(Mesh));
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

    cudaStatus = cudaMalloc((void**)&gpu_scene.dev_meshes, scene.meshCount * sizeof(Mesh));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Meshes cudaMalloc failed!");
    }
    cudaStatus = cudaMemcpy(gpu_scene.dev_meshes, h_mesh, scene.meshCount * sizeof(Mesh), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Meshes cudaMemcpy failed!");
    }
    cudaStatus = cudaMalloc((void**)&gpu_scene.dev_materials, scene.materialCount * sizeof(Material));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Materials cudaMalloc failed!");
    }
    cudaStatus = cudaMemcpy(gpu_scene.dev_materials, &scene.materials[0], scene.materialCount * sizeof(Material), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Materials cudaMemcpy failed!");
    }

    gpu_scene.materialCount = scene.materialCount;
    gpu_scene.meshCount = scene.meshCount;

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

cudaError_t trace(Vector3f** dev_out, const Camera& camera, uint width, uint height, const GPU_Scene &scene, curandState* d_state) {
    cudaError_t cudaStatus;

    unsigned int blockSize = NUM_THREADS;
    unsigned int gridSize = (width * height) / NUM_THREADS + ((width * height) % NUM_THREADS == 0 ? 0 : 1);

    Vector3f cz = normalise(camera.direction);
    Vector3f cy(0, 1, 0);
    Vector3f cx = cross(cz, cy).normalise();
    cy = cross(cx, cz);
    Basis basis = { cx, cy, cz };

    // Launch a kernel on the GPU with one thread for each element.
    traceKernel << <gridSize, blockSize >> >(*dev_out, width, height, camera, basis, scene, d_state);

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
