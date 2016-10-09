#include "Scene.h"

Scene::Scene()
{
}


Scene::~Scene()
{
}

Mesh &Scene::getMesh(unsigned int i) {
	return meshes[i];
}

void Scene::addMesh(Mesh &mesh) {
	meshes.push_back(mesh);
}
