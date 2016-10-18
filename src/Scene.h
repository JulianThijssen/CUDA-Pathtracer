#pragma once
#ifndef SCENE_H
#define SCENE_H

#define CUDA __host__ __device__
#define CAMERA_FAR 10000
#define EPSILON 0.001f

#include "cuda_runtime.h"
#include "Ray.h"
#include "Mesh.h"
#include "Material.h"
#include "Vector3f.h"

#include <vector>

class Scene
{
public:
	CUDA Scene();
	CUDA ~Scene();

	// Host
	Mesh &getMesh(unsigned int i);
	void addMesh(Mesh &mesh);
	std::vector<Mesh> meshes;
	std::vector<Material> materials;

	// Device
	__device__ inline bool Scene::intersect(const Ray &ray, Mesh **mesh, float &t, Vector3f &n) const {
		t = CAMERA_FAR;
		bool hit = false;

		for (unsigned int j = 0; j < meshCount; j++) {
			Vector3f mesh_n(0, 0, 0);

			float intersect_t = dev_meshes[j].intersect(ray.o + ray.d*EPSILON, ray.d, mesh_n);
			if (intersect_t > 0 && intersect_t < t) {
				t = intersect_t;
				n.set(mesh_n);
				*mesh = &dev_meshes[j];
				hit = true;
			}
		}
		return hit;
	}

	Mesh *dev_meshes;
	Material *dev_materials;

	// Both
	unsigned int meshCount;
	unsigned int materialCount;
private:

};

#endif /* SCENE_H */
