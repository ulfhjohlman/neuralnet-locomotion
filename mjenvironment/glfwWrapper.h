#pragma once
#include "glfw3.h"
#include <stdexcept>
#include <atomic>

struct WindowRectangle                   // OpenGL rectangle
{
	int left;                       // left (usually 0)
	int bottom;                     // bottom (usually 0)
	int width;                      // width (usually buffer width)
	int height;                     // height (usually buffer height)
};

class glfwWrapper 
{
public:
	static bool initializeGLFW() {
		if (!bInitialized.exchange(true)) {
			if (!glfwInit())
				throw std::runtime_error("glfw not initialized successfully.");
		}
	}

	glfwWrapper() : window(nullptr) {
		// init GLFW
		glfwWrapper::initializeGLFW();
		instances.fetch_add(1);

		// create window, make OpenGL context current, request v-sync
		window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
		glfwMakeContextCurrent(window);
		glfwSwapInterval(1);
	}
	~glfwWrapper() { 
		instances.fetch_sub(1);
	}
	GLFWwindow* getWindow() {
		return window;
	}

	void installKeyCallback(void (*callback)(GLFWwindow*, int , int , int , int )) {
		glfwSetKeyCallback(window, callback);
	}

	void installMouseCallback(void(*callback)(GLFWwindow*, int, int, int)) {
		glfwSetMouseButtonCallback(window, callback);
	}

	void installCursorPosCallback(void(*callback)(GLFWwindow*, double, double )) {
		glfwSetCursorPosCallback(window, callback);
	}

	void installScrollPosCallback(void(*callback)(GLFWwindow*, double, double)) {
		glfwSetScrollCallback(window, callback);
	}
	
private:
	static std::atomic_bool bInitialized;
	static std::atomic_int instances;
	GLFWwindow* window;
};

std::atomic_bool glfwWrapper::bInitialized = false; //Init class check
std::atomic_int glfwWrapper::instances = 0;