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
	std::vector<Mesh> &getMeshes();
	Mesh &getMesh(unsigned int i);
	void addMesh(Mesh &mesh);
	unsigned int meshCount();
private:
	std::vector<Mesh> meshes;
};

#endif /* SCENE_H */
