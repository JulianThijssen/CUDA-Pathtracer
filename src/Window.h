/* ---------------------------------------------------------------------------
** This software is in the public domain, furnished "as is", without technical
** support, and with no warranty, express or implied, as to its usefulness for
** any purpose.
**
** Window.h
** Declares a window class containing basic window creation functionality
** and initialization of an OpenGL context through GLFW.
**
** Author: Julian Thijssen
** -------------------------------------------------------------------------*/

#pragma once
#ifndef WINDOW_H
#define WINDOW_H

#include <string>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Window {
public:
	Window();
	Window(const char* title);
	Window(const char* title, int width, int height);
	
	std::string getTitle();
	void setTitle(std::string title);
	void getSize(unsigned int& w, unsigned int& h);
	void setSize(int width, int height);
	void update();
	void close();
	bool isClosed();
private:
	GLFWwindow* window;
	std::string title;
	int width;
	int height;

	void destroy();
	static void onError(int error, const char* description);
};

#endif /* WINDOW_H */
