#pragma once
#ifndef MESH_H
#define MESH_H

#include "Vector3f.h"

#include <curand.h>
#include <curand_kernel.h>

class Face {
public:
	unsigned int v0, v1, v2;
	unsigned int n0, n1, n2;
};

class Mesh {
public:
	unsigned int materialIndex;
	unsigned int numVerts;
	unsigned int numNorms;
	unsigned int numFaces;
	Vector3f* vertices;
	Vector3f* normals;
	Face* faces;

	~Mesh() {
		//delete vertices;
		//delete normals;
		//delete faces;
	}

	__device__ inline Vector3f getRandomSample(const unsigned int idx, curandState *state)
	{
		unsigned int faceIndex = (unsigned int)(curand_uniform(&state[idx]) * numFaces);

		Face face = faces[faceIndex];

		Vector3f v0 = vertices[face.v0];
		Vector3f v1 = vertices[face.v1];
		Vector3f v2 = vertices[face.v2];

		float alpha = curand_uniform(&state[idx]);
		float beta = curand_uniform(&state[idx]);
		if (alpha + beta >= 1) {
			alpha = 1 - alpha;
			beta = 1 - beta;
		}
		float gamma = 1 - alpha - beta;
		Vector3f sample = v0 * alpha + v1 * beta + v2 * gamma;

		return sample;
	}

	__device__ inline float intersect(const Vector3f o, const Vector3f& d, Vector3f& n)
	{
		float epsilon = 0.000001f;

		float minT = FLT_MAX;

		bool intersect = false;

		for (unsigned int i = 0; i < numFaces; i++) {
			Face face = faces[i];

			Vector3f v0 = vertices[face.v0];
			Vector3f v1 = vertices[face.v1];
			Vector3f v2 = vertices[face.v2];

			Vector3f n0 = normals[face.n0];

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
};

#endif /* MESH_H */
