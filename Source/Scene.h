#pragma once
#ifndef SCENE_H
#define SCENE_H

#define CUDA __host__ __device__

#include "cuda_runtime.h"
#include "Ray.h"
#include "Mesh.h"
#include "Material.h"
#include "Vector3f.h"

#include <vector>

CUDA struct HitInfo {
	bool hit;
	float t;
	Vector3f n;
	Mesh *mesh;
};

class Scene
{
public:
	Scene();
	~Scene();

	Mesh* getMesh(unsigned int i);
	void addMesh(Mesh *mesh);
	std::vector<Mesh*> meshes;
	std::vector<Material> materials;

	unsigned int meshCount;
	unsigned int materialCount;
private:

};

class GPU_Scene
{
public:
    CUDA GPU_Scene();
    CUDA ~GPU_Scene();

    // Device
    __device__ HitInfo intersect(const Ray &ray) const {
        HitInfo info = { false, 0, Vector3f(), 0 };
        info.t = 10000;
        
        for (unsigned int j = 0; j < meshCount; j++) {
            Vector3f n(0, 0, 0);
        
            float t = dev_meshes[j].intersect(ray.o + ray.d*0.0001, ray.d, n);
            if (t > 0 && t < info.t) {
                info.t = t;
                info.n.set(n);
                info.mesh = &dev_meshes[j];
                info.hit = true;
            }
        }
        return info;
    }

    Mesh *dev_meshes;
    Material *dev_materials;

    unsigned int meshCount;
    unsigned int materialCount;
};

#endif /* SCENE_H */
