#include "mjEnvironment.h"
#include "GeneticAlgorithm.h"

const int g_population_size = 128;

mjEnvironment* environment;
GeneticAlgorithm* ga;

// interaction
bool b_render = true;
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;

GLFWwindow* window = nullptr;
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods);
void mouse_button(GLFWwindow* window, int button, int act, int mods);
void mouse_move(GLFWwindow* window, double xpos, double ypos);
void scroll(GLFWwindow* window, double xoffset, double yoffset);
bool init();

int main() {
	if (!init())
		return 1;

	environment  = new mjEnvironment(g_population_size);
	int nsensors = environment->m_model->nsensordata;
	int nctrls	 = environment->m_model->nu;
	ga			 = new GeneticAlgorithm(g_population_size, nsensors, nctrls);

	auto objective = [](mjModel const* m, mjData* d) { return d->site_xpos[2] + d->site_xpos[0] - std::abs(d->site_xpos[1]); };
	environment->setObjective(objective);

	ga->setEnvironment(environment);
	
	while (!glfwWindowShouldClose(window))
	{
		ga->run();
		//render here
		if (b_render) {
			// get framebuffer viewport
			mjrRect viewport = { 0, 0, 0, 0 };
			glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

			

			environment->render(viewport);

			// swap OpenGL buffers (blocking call due to v-sync)
			glfwSwapBuffers(window);
		}

		// process pending GUI events, call GLFW callbacks
		glfwPollEvents();
	}

	return 0;
}

bool init() {
	// init GLFW
	if (!glfwInit()) {
		mju_error("Could not initialize GLFW");
		return false;
	}

	// create window, make OpenGL context current, request v-sync
	window = glfwCreateWindow(1200, 900, "environment", NULL, NULL);
	if (!window) {
		glfwTerminate();
		mju_error("window could not init.");
		return false;
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glfwSetKeyCallback(window, keyboard);
	glfwSetCursorPosCallback(window, mouse_move);
	glfwSetMouseButtonCallback(window, mouse_button);
	glfwSetScrollCallback(window, scroll);

	return true;
}


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
	// backspace: reset simulation
	if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
	{
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_Q) {
		//ga->mutate();
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_U) {
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_R)
	{
		b_render = !b_render;
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_W)
	{
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_S)
	{
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_A)
	{
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_D)
	{
	}


}


// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
	// update button state
	button_left = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS);
	button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS);
	button_right = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS);

	// update mouse position
	glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
	// no buttons down: nothing to do
	if (!button_left && !button_middle && !button_right)
		return;

	// compute mouse displacement, save
	double dx = xpos - lastx;
	double dy = ypos - lasty;
	lastx = xpos;
	lasty = ypos;

	// get current window size
	int width, height;
	glfwGetWindowSize(window, &width, &height);

	// get shift key state
	bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
		glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);

	// determine action based on mouse button
	mjtMouse action;
	if (button_right)
		action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
	else if (button_left)
		action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
	else
		action = mjMOUSE_ZOOM;

	// move camera
	if(environment)
		mjv_moveCamera(environment->m_model, action, dx / height, dy / height, &environment->scn, &environment->cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
	// emulate vertical mouse motion = 5% of window height
	if (environment)
		mjv_moveCamera(environment->m_model, mjMOUSE_ZOOM, 0, -0.05*yoffset, &environment->scn, &environment->cam);
}
