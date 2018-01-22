#pragma once
#ifndef MATHS_H
#define MATHS_H

#define PI 3.14159265
#define ONE_OVER_PI 0.318309886
#define EPSILON 0.001

__device__ float gmin(float a, float b) {
    return a < b ? a : b;
}

__device__ float gmax(float a, float b) {
    return a > b ? a : b;
}

__device__ Vector3f mix(Vector3f x, Vector3f y, float a) {
    return x * (1 - a) + y * a;
}

__device__ float clamp(float x, float low, float high) {
    return gmin(gmax(x, low), high);
}

#endif /* MATHS_H */
