#pragma once

#include "Vector3f.h"

#define CUDA __host__ __device__

class Ray {
public:
	CUDA Ray(Vector3f o, Vector3f d) : o(o), d(d) { }

	Vector3f o;
	Vector3f d;
};
