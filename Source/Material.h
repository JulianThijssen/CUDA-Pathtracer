#pragma once
#ifndef MATERIAL_H
#define MATERIAL_H


#include "Vector3f.h"

class Material {
public:
	Material() { }

	Material(Vector3f albedo, Vector3f emission) :
		albedo(albedo),
		emission(emission) { }

	Vector3f albedo;
	Vector3f emission;
};

#endif /* MATERIAL_H */
