#include "mjEnvironment.h"
#include "GeneticAlgorithm.h"
#include "RandomEngineFactory.h"

#include <stdexcept>

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

void setup() {
	RandomEngineFactory::initialize();

	environment = new mjEnvironment(g_population_size);
	if (!environment)
		throw std::runtime_error("Could not make environment.");
	int nsensors = environment->m_model->nsensordata;
	int nctrls = environment->m_model->nu;
	ga = new GeneticAlgorithm(g_population_size, nsensors, nctrls);
	if (!ga)
		throw std::runtime_error("Could not start make GA.");

	//auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[5] - 0.1*std::abs(d->site_xpos[3]); };
	auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[2] + 1.2*d->site_xpos[0] - 0.45 * std::abs(d->site_xpos[1]); };
	//auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[2]; };
	environment->setObjective(objective);

	ga->setEnvironment(environment);
}

int main() {
	if (!init())
		return 1;
	
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

	if (environment)
		delete environment;
	if (ga)
		delete ga;

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

	try {
		setup();
	}
	catch (NeuralNetException e) {
		std::cerr << e.what() << std::endl;
		return false;
	}
	catch (std::runtime_error e) {
		std::cerr << e.what() << std::endl;
		return false;
	}
	catch (std::bad_alloc e) {
		std::cerr << "bad alloc" << std::endl;
		return false;
	}

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
		g_simulation_steps += 2;
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_U) {
		g_simulation_steps -= 2;
		if (g_simulation_steps < 0)
			g_simulation_steps = 0;
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
		std::string input;
		std::cout << "Type in the max simulation time: ";
		std::cin >> input;
		try {
			g_max_simulation_time = std::stof(input);
		}
		catch (...)
		{
			std::cout << "Invalid value, try again\n";
		}
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_C)
	{
		std::string input;
		std::cout << "Type in the crossover probability: ";
		std::cin >> input;
		try {
			g_crossover_probability = std::stof(input);
		}
		catch (...)
		{
			std::cout << "Invalid value, try again\n";
		}
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_D)
	{
		std::string input;
		std::cout << "Type in the generation number (mutation rate): ";
		std::cin >> input;
		try {
			ga->m_generation = std::stoi(input);
		}
		catch (...)
		{
			std::cout << "Invalid value, try again\n";
		}
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_P)
	{
		std::string input;
		std::cout << "Type in the kill height (0.95 default): ";
		std::cin >> input;
		try {
			g_minimum_kill_height = std::stof(input);
		}
		catch (...)
		{
			std::cout << "Invalid value, try again\n";
		}
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
