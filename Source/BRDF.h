/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** BRDF.h
** Declares bi-directional reflectance functions
**
** Author: Julian Thijssen
** -------------------------------------------------------------------------*/

#pragma once
#ifndef BRDF_H
#define BRDF_H

#include "cuda_runtime.h"

#include "Vector3f.h"
#include "Material.h"
#include "Maths.h"

/* Calculates the diffuse contribution of the light */
__device__ float CosTheta(Vector3f N, Vector3f L) {
    return gmax(0, dot(N, L));
}

__device__ Vector3f Fresnel(Vector3f BaseColor, float Metalness, float b) {
    Vector3f F0 = mix(Vector3f(0.04), BaseColor, Metalness);
    return F0 + (Vector3f(1.0) - F0) * pow(1.0 - b, 5.0);
}

__device__ float GGX(float NdotH, float Roughness) {
    float a = Roughness * Roughness;
    float a2 = a * a;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

__device__ float Schlick(float NdotL, float NdotV, float Roughness) {
    float a = Roughness + 1.0;
    float k = a * a * 0.125;
    float G1 = NdotL / (NdotL * (1.0 - k) + k);
    float G2 = NdotV / (NdotV * (1.0 - k) + k);
    return G1 * G2;
}

__device__ Vector3f Lambert(Vector3f Kd) {
    return Kd / PI;
}

__device__ Vector3f CookTorrance(Vector3f N, Vector3f V, Vector3f H, Vector3f L, Vector3f BaseColor, float Metalness, float Roughness) {
    float NdotH = gmax(0.0f, dot(N, H));
    float NdotV = gmax(1e-7f, dot(N, V));
    float NdotL = gmax(1e-7f, dot(N, L));
    float VdotH = gmax(0.0f, dot(V, H));

    float D = GGX(NdotH, Roughness);
    float G = Schlick(NdotL, NdotV, Roughness);
    Vector3f F = Fresnel(BaseColor, Metalness, VdotH);

    return (F * D * G) / (4.0 * NdotL * NdotV);
}

__device__ Vector3f BRDF(Vector3f N, Vector3f L, Material& material) {
    float cos = CosTheta(N, L);
    Vector3f LambertBRDF = Lambert(material.albedo);
    //Vector3f CookBRDF = CookTorrance(info.n, V, H, L, mat.albedo, 0, 1);

    return (LambertBRDF);
}

#endif /* BRDF_H */
