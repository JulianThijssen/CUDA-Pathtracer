#pragma once
#ifndef SCENE_H
#define SCENE_H

#include "Mesh.h"
#include <vector>

class Scene
{
public:
	Scene();
	~Scene();

	// Host
	Mesh &getMesh(unsigned int i);
	void addMesh(Mesh &mesh);
	std::vector<Mesh> meshes;

	// Device
	__device__ bool intersect(Mesh &mesh, float &t);
	Mesh *dev_meshes;

	// Both
	unsigned int meshCount;
private:

};

#endif /* SCENE_H */
