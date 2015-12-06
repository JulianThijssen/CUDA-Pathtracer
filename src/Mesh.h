#pragma once
#ifndef MESH_H
#define MESH_H

#include "Vector3f.h"

class Face {
public:
	unsigned int v0, v1, v2;
	unsigned int n0, n1, n2;
};

class Mesh {
public:
	float emission = 0;
	Vector3f albedo;
	unsigned int numVerts;
	unsigned int numNorms;
	unsigned int numFaces;
	Vector3f* vertices;
	Vector3f* normals;
	Face* faces;
};

__device__ inline float intersect(const Vector3f o, const Vector3f& d, const Mesh& m, Vector3f& n)
{
	float epsilon = 0.000001f;

	float minT = FLT_MAX;

	bool intersect = false;

	for (unsigned int i = 0; i < m.numFaces; i++) {
		Face face = m.faces[i];

		Vector3f v0 = m.vertices[face.v0];
		Vector3f v1 = m.vertices[face.v1];
		Vector3f v2 = m.vertices[face.v2];

		Vector3f n0 = m.normals[face.n0];

		Vector3f e1, e2;
		Vector3f P, Q, T;
		float det, invDet, u, v;
		float t;

		// Find the edge vectors
		e1 = v1 - v0;
		e2 = v2 - v0;

		// Calculate determinant
		P = cross(d, e2);
		det = dot(e1, P);

		// If det near zero, ray lies in plane of triangle
		if (det > -epsilon && det < epsilon) {
			continue;
		}

		invDet = 1.0f / det;

		// Calculate distance from v0 to ray origin
		T = o - v0;

		// Calculate the u parameter and test bound
		u = dot(T, P) * invDet;
		if (u < 0 || u > 1) {
			// The intersection lies outside of the triangle
			continue;
		}

		// Calculate the v parameter and test bound
		Q = cross(T, e1);

		v = dot(d, Q) * invDet;
		if (v < 0 || u + v > 1) {
			// The intersection lies outside of the triangle
			continue;
		}

		t = dot(e2, Q) * invDet;

		if (t > epsilon && t < minT) {
			minT = t;
			n.set(n0.x, n0.y, n0.z);
			intersect = true;
		}
	}

	if (intersect) {
		return minT;
	}

	return 0;
}

#endif /* MESH_H */
