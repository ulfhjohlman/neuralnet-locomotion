#include "mjEnvironment.h"
#include "GeneticAlgorithm.h"
#include "RandomEngineFactory.h"
#include "ModelSetup.h"

#include <stdexcept>

mjEnvironment* environment = nullptr;
GeneticAlgorithm* ga = nullptr;

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

void setup_ant() {
	environment = new mjEnvironment(g_population_size, 128, "ant.xml", "environment.xml");
	if (!environment)
		throw std::runtime_error("Could not make environment.");
	int nsensors = environment->m_model->nsensordata;
	int nctrls = environment->m_model->nu;
	int nrecurrent = 16;
	ga = new GeneticAlgorithm(g_population_size, nsensors, nctrls);
	if (!ga)
		throw std::runtime_error("Could not start make GA.");

	auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[2] + 1.2*d->site_xpos[0] - 0.45 * std::abs(d->site_xpos[1]); };
	environment->setObjective(objective);

	ScalingLayer input_scaling(nsensors + nrecurrent, 1);
	const ScalarType scale_touch = 1.0 / 500.0;
	input_scaling(0) = scale_touch;
	input_scaling(1) = scale_touch;
	input_scaling(2) = scale_touch;
	input_scaling(3) = scale_touch;
	//input_scaling(4) = scale_touch;

	const ScalarType scale_acc = 1.0 / 10.0;
	input_scaling(5) = scale_acc;
	input_scaling(6) = scale_acc;
	input_scaling(7) = scale_acc;

	//gyro 8 9 10

	const ScalarType range_finder = 1.0 / 10.0;
	input_scaling(11) = range_finder;
	input_scaling(12) = range_finder;
	input_scaling(13) = range_finder;

	//rest joint pos/vel

	environment->setScalingLayer(input_scaling);
	ga->setEnvironment(environment);
}

void setup_humanoid() {
	environment = new mjEnvironment(g_population_size, 128, "humanoid.xml", "environment.xml");
	if (!environment)
		throw std::runtime_error("Could not make environment.");
	int nsensors = environment->m_model->nsensordata;
	int nctrls = environment->m_model->nu;
	int nrecurrent = 32;
	ga = new GeneticAlgorithm(g_population_size, nsensors, nctrls);
	if (!ga)
		throw std::runtime_error("Could not start make GA.");

	auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[2] - 0.9*( d->site_xpos[2] < 0.9 ); };
	environment->setObjective(objective);

	ScalingLayer input_scaling(nsensors + nrecurrent, 1);
	const ScalarType scale_touch = 1.0 / 1000.0;
	input_scaling(0) = scale_touch;
	input_scaling(1) = scale_touch;
	input_scaling(2) = scale_touch;
	input_scaling(3) = scale_touch;
	input_scaling(4) = scale_touch;
	input_scaling(5) = scale_touch;

	const ScalarType scale_acc = 1.0 / 10.0;
	input_scaling(6) = scale_acc;
	input_scaling(7) = scale_acc;
	input_scaling(8) = scale_acc;

	//gyro 9 10 11

	//rest joint pos/vel

	environment->setScalingLayer(input_scaling);
	ga->setEnvironment(environment);
}

void setup_swimmer() {
	environment = new mjEnvironment(g_population_size, 128, "swimmer.xml", "environment.xml");
	if (!environment)
		throw std::runtime_error("Could not make environment.");
	int nsensors = environment->m_model->nsensordata;
	int nctrls = environment->m_model->nu;
	int nrecurrent = 16;
	ga = new GeneticAlgorithm(g_population_size, nsensors, nctrls);
	if (!ga)
		throw std::runtime_error("Could not start make GA.");

	auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[0] * d->site_xpos[0]; };
	environment->setObjective(objective);

	ScalingLayer input_scaling(nsensors + nrecurrent, 1);

	environment->setScalingLayer(input_scaling);
	ga->setEnvironment(environment);
}

void setup_walker2d() {
	environment = new mjEnvironment(g_population_size, 128, "walker2d.xml", "environment.xml");
	if (!environment)
		throw std::runtime_error("Could not make environment.");
	int nsensors = environment->m_model->nsensordata;
	int nctrls = environment->m_model->nu;
	int nrecurrent = 16;
	ga = new GeneticAlgorithm(g_population_size, nsensors, nctrls);
	if (!ga)
		throw std::runtime_error("Could not start make GA.");

	auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[0] + 1.5 * d->qvel[0] + 0.1*d->site_xpos[2]; };
	environment->setObjective(objective);

	ScalingLayer input_scaling(nsensors + nrecurrent, 1);
	input_scaling.getScaling().array() /= 2;
	const ScalarType scale_touch = 1.0 / 1000.0;
	input_scaling(0) = scale_touch;
	input_scaling(1) = scale_touch;
	input_scaling(2) = scale_touch;
	input_scaling(3) = scale_touch;


	environment->setScalingLayer(input_scaling);
	ga->setEnvironment(environment);
}


void setup_invdoublepole() {
	environment = new mjEnvironment(g_population_size, 128, "invdoublependulum2D.xml", "environment.xml");
	if (!environment)
		throw std::runtime_error("Could not make environment.");
	int nsensors = environment->m_model->nsensordata;
	int nctrls = environment->m_model->nu;
	int nrecurrent = 0;
	ga = new GeneticAlgorithm(g_population_size, nsensors, nctrls);
	if (!ga)
		throw std::runtime_error("Could not start make GA.");

	auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[5] - 0.1*std::abs(d->site_xpos[3]) - 0.1*std::abs(d->site_xpos[4]); };
	environment->setObjective(objective);

	ScalingLayer input_scaling(nsensors + nrecurrent, 1);

	environment->setScalingLayer(input_scaling);
	ga->setEnvironment(environment);
}

void setup_hopper() {
	environment = new mjEnvironment(g_population_size, 128, "hopper.xml", "environment.xml");
	if (!environment)
		throw std::runtime_error("Could not make environment.");
	int nsensors = environment->m_model->nsensordata;
	int nctrls = environment->m_model->nu;
	int nrecurrent = 16;
	ga = new GeneticAlgorithm(g_population_size, nsensors, nctrls);
	if (!ga)
		throw std::runtime_error("Could not start make GA.");

	auto objective = [](mjModel const* m, mjData* d) { return 1.0*d->site_xpos[0] + .3*d->site_xpos[2] + 0.01; };
	environment->setObjective(objective);

	ScalingLayer input_scaling(nsensors + nrecurrent, 1);
	const ScalarType scale_touch = 1.0 / 1000.0;
	input_scaling(0) = scale_touch;
	input_scaling(1) = scale_touch;

	environment->setScalingLayer(input_scaling);
	ga->setEnvironment(environment);
}


int main() {
	RandomEngineFactory::initialize();

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
		//setup();
		//setup_ant();
		//setup_humanoid();
		//setup_invdoublepole();
		//setup_swimmer();
		setup_walker2d();
		//setup_hopper();
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


float read_float_input(const char* message)
{
	std::string input;
	float res = 0;
	std::cout << message;
	std::cin >> input;
	read_again:
	try { res = std::stof(input); }
	catch (...) {
		std::cout << "Invalid value, try again\n";
		goto read_again;
	}
	return res;
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
		g_max_simulation_time = read_float_input("max simulation time");
		if (g_max_simulation_time < 0.1)
			g_max_simulation_time = 0.1;
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_C)
	{
		g_crossover_probability = read_float_input("Type in the crossover probability: ");
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
		g_minimum_kill_height = read_float_input("Type in the kill height (0.95 default): ");
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
