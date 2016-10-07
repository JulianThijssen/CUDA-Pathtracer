#include "Scene.h"

Scene::Scene()
{
}


Scene::~Scene()
{
}

std::vector<Mesh> &Scene::getMeshes() {
	return meshes;
}

Mesh &Scene::getMesh(unsigned int i) {
	return meshes[i];
}

void Scene::addMesh(Mesh &mesh) {
	meshes.push_back(mesh);
}

unsigned int Scene::meshCount() {
	return meshes.size();
}
