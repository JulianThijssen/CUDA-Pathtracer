#include "Scene.h"

#define CAMERA_FAR 10000

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

__device__ bool Scene::intersect(Mesh &mesh, float &t) {
	//// Scene intersection
	//float closest_t = CAMERA_FAR;
	//Vector3f hit_n;
	//Mesh* mesh;

	//for (int j = 0; j < meshCount; j++) {
	//	Vector3f n(0, 0, 0);

	//	float t = intersect(o + d*EPSILON, d, meshes[j], n);
	//	if (t > 0 && t < closest_t) {
	//		closest_t = t;
	//		hit_n.set(n.x, n.y, n.z);
	//		mesh = &meshes[j];
	//	}
	//}
}
