#include "Window.h"

#include "Scene.h"
#include "ModelLoader.h"
#include "Camera.h"
#include "CudaRenderer.h"

#include <glad/glad.h>

#include <ctime>
#include <iostream>

int main()
{
	Window window("Path Tracer", 512, 512);

    CudaRenderer renderer;

    float rotation = 0;
    float time = 0;

	unsigned int width, height;
	window.getSize(width, height);

    // Upload the scene
    Scene scene;
    loadScene(scene, std::string("res/cornell_box.obj"));

    renderer.init(scene);
    renderer.resize(Size(width, height));

	clock_t begin = clock();
	int frames = 0;
	int iterations = 0;

	while (!window.isClosed()) {
        //d.set(sin(rotation), 0, cos(rotation));

        //if (iterations == 1) {
        //    time += 0.03f;
        //    rotation = sin(time) * 0.1f;
        //    iterations = 0;
        //    // Clear the accumulated buffer
        //    for (unsigned int i = 0; i < width * height; i++) {
        //        out[i] = 0;
        //    }
        //}

		clock_t end = clock();
		double elapsed = double(end - begin) / CLOCKS_PER_SEC;
		if (elapsed >= 1) {
			begin = clock();
			printf("Frames: %d\n", frames);
			frames = 0;
		}
		
        renderer.update();

		glDrawPixels(width, height, GL_RGB, GL_FLOAT, renderer.final);

		window.update();
		frames++;
	}

    renderer.destroy();

	return 0;
}
