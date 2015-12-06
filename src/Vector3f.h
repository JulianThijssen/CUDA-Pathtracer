/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Vector3f.h
** Declares a vector consisting of 3 float values and its helper functions
**
** Author: Julian Thijssen
** -------------------------------------------------------------------------*/

#pragma once
#ifndef VECTOR3F_H
#define VECTOR3F_H

#include "cuda_runtime.h"
#include <string>
#include <cmath>
#include <sstream>

#define CUDA __host__ __device__

class Vector3f {
public:
	float x, y, z;

	static const Vector3f ZERO;
	static const Vector3f FORWARD;
	static const Vector3f UP;

	/* Core */
	CUDA Vector3f() : x(0), y(0), z(0) {}
	CUDA Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
	CUDA void set(float x, float y, float z) {
		this->x = x;
		this->y = y;
		this->z = z;
	}
	CUDA void set(const Vector3f& v) {
		x = v.x;
		y = v.y;
		z = v.z;
	}
	CUDA Vector3f& normalise() {
		float l = length();
		x /= l;
		y /= l;
		z /= l;
		return *this;
	}

	CUDA float length() const {
		return sqrt(x * x + y * y + z * z);
	}
	CUDA std::string str() const {
		std::stringstream ss;
		ss << "(" << x << ", " << y << ", " << z << ")";
		return ss.str();
	}

	/* Operator overloads */
	CUDA bool operator==(const Vector3f& v) const {
		return x == v.x && y == v.y && z == v.z;
	}
	CUDA bool operator!=(const Vector3f& v) const {
		return x != v.x || y != v.y || z != v.z;
	}
	CUDA Vector3f& operator+=(const Vector3f& v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}

	CUDA Vector3f& operator-=(const Vector3f& v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	CUDA Vector3f& operator*=(const Vector3f& v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}
	CUDA Vector3f& operator*=(float scale) {
		x *= scale;
		y *= scale;
		z *= scale;
		return *this;
	}
	CUDA Vector3f& operator/=(const Vector3f& v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
		return *this;
	}
	CUDA Vector3f operator+(const Vector3f& v) const {
		return Vector3f(x + v.x, y + v.y, z + v.z);
	}
	CUDA Vector3f operator-(const Vector3f& v) const {
		return Vector3f(x - v.x, y - v.y, z - v.z);
	}
	CUDA Vector3f operator-() const {
		return Vector3f(-x, -y, -z);
	}
	CUDA Vector3f operator*(float scale) const {
		return Vector3f(x * scale, y * scale, z * scale);
	}
	CUDA Vector3f operator/(float divisor) const {
		return Vector3f(x / divisor, y / divisor, z / divisor);
	}
};

const Vector3f ZERO = Vector3f(0, 0, 0);
const Vector3f FORWARD = Vector3f(0, 0, -1);
const Vector3f UP = Vector3f(0, 1, 0);

/* Utility functions */
CUDA inline float dot(const Vector3f& v1, const Vector3f& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
CUDA inline Vector3f cross(const Vector3f& v1, const Vector3f& v2) {
	Vector3f v;
	v.x = v1.y * v2.z - v1.z * v2.y;
	v.y = v2.x * v1.z - v2.z * v1.x;
	v.z = v1.x * v2.y - v1.y * v2.x;
	return v;
}
CUDA inline Vector3f negate(const Vector3f& v) {
	return Vector3f(-v.x, -v.y, -v.z);
}
CUDA inline Vector3f normalise(const Vector3f& v) {
	float l = v.length();
	return Vector3f(v.x / l, v.y / l, v.z / l);
}

#endif /* VECTOR3F_H */
