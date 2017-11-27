#pragma once
#include "LayeredNeuralNet.h"
#include "LayeredTopology.h"


#include "mujoco.h"
#include <iostream>
#include <memory>

#include "Generator.h"
#include "glfw_helper.h"


int n_inputs = 0;
int n_outputs = 0;
LayeredNeuralNet* controller = nullptr;
LayeredNeuralNet* feedbacknn = nullptr;
MatrixType state;
bool controller_on = false;

const char* model_name = "invdoublependulum2D.xml";


// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data


mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

mjvFigure figsensor; //sensor figure

					 // mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;
int sign = 1;

void createNeuralController() {
	if (!m)
		throw std::runtime_error("init model first");

	controller = new LayeredNeuralNet; //memory is managed by network
	controller->load("generation4000/_0walker2dRewardEng3");
	//controller->clearInternalStates();

	int N_layers = controller->getTopology()->getNumberOfLayers();
	n_inputs = controller->getTopology()->getLayerSize(0);
	n_outputs = controller->getTopology()->getLayerSize(N_layers-1);

	std::cout << n_inputs - m->nsensordata << std::endl;
	std::cout << n_outputs - m->nu << std::endl;

	state.resize(n_inputs, 1);
}

void destroyNeuralController() {
	if (controller)
		delete controller;
	controller = nullptr;
}

// load mjb or xml model
void loadmodel(const char* filename)
{
	// make sure filename is given
	if (!filename)
		return;

	// load and compile
	char error[1000] = "could not load binary model";
	mjModel* mnew = 0;
	if (strlen(filename) > 4 && !strcmp(filename + strlen(filename) - 4, ".mjb"))
		mnew = mj_loadModel(filename, 0);
	else
		mnew = mj_loadXML(filename, 0, error, 1000);
	if (!mnew) {
		printf("%s\n", error);
		using namespace std::chrono_literals;
		return;
	}

	// delete old model, assign new
	mj_deleteData(d);
	mj_deleteModel(m);
	m = mnew;
	m->qpos0[0] = -1; //Center rectangle
	m->qpos0[1] = -1;
	d = mj_makeData(m);
	mj_forward(m, d);
}

// init sensor figure
void sensorinit(void)
{
	// set figure to default
	mjv_defaultFigure(&figsensor);

	// set flags
	figsensor.flg_extend = 1;
	figsensor.flg_barplot = 1;

	// title
	strcpy(figsensor.title, "Sensor data");

	// y-tick nubmer format
	strcpy(figsensor.yformat, "%.0f");

	// grid size
	figsensor.gridsize[0] = 2;
	figsensor.gridsize[1] = 3;

	// minimum range
	figsensor.range[0][0] = 0;
	figsensor.range[0][1] = 0;
	figsensor.range[1][0] = -1;
	figsensor.range[1][1] = 1;
}

// update sensor figure
void sensorupdate(void)
{
	static const int maxline = 10;

	// clear linepnt
	for (int i = 0; i < maxline; i++)
		figsensor.linepnt[i] = 0;

	// start with line 0
	int lineid = 0;

	// loop over sensors
	for (int n = 0; n < m->nsensor; n++)
	{
		// go to next line if type is different
		if (n > 0 && m->sensor_type[n] != m->sensor_type[n - 1])
			lineid = mjMIN(lineid + 1, maxline - 1);

		// get info about this sensor
		mjtNum cutoff = (m->sensor_cutoff[n] > 0 ? m->sensor_cutoff[n] : 1);
		int adr = m->sensor_adr[n];
		int dim = m->sensor_dim[n];

		// data pointer in line
		int p = figsensor.linepnt[lineid];

		// fill in data for this sensor
		for (int i = 0; i < dim; i++)
		{
			// check size
			if ((p + 2 * i) >= mjMAXLINEPNT / 2)
				break;

			// x
			figsensor.linedata[lineid][2 * p + 4 * i] = (float)(adr + i);
			figsensor.linedata[lineid][2 * p + 4 * i + 2] = (float)(adr + i);

			// y
			figsensor.linedata[lineid][2 * p + 4 * i + 1] = 0;
			figsensor.linedata[lineid][2 * p + 4 * i + 3] = (float)(state(adr + i) / cutoff);
		}

		// update linepnt
		figsensor.linepnt[lineid] = mjMIN(mjMAXLINEPNT - 1,
			figsensor.linepnt[lineid] + 2 * dim);
	}
}

// show sensor figure
void sensorshow(mjrRect rect)
{
	// render figure on the right
	mjrRect viewport = { rect.width - rect.width / 4, rect.bottom, rect.width / 4, rect.height / 3 };
	mjr_figure(viewport, &figsensor, &con);
}



// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
	// backspace: reset simulation
	if (act == GLFW_PRESS && key == GLFW_KEY_BACKSPACE)
	{
		mj_resetData(m, d);
		mj_forward(m, d);
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_Q) {
		controller_on = !controller_on;
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_U) {
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_R)
	{
		// load and compile model
		loadmodel(model_name);
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
	mjv_moveCamera(m, action, dx / height, dy / height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
	// emulate vertical mouse motion = 5% of window height
	mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}


// simple controller applying damping to each dof
void test_controller(const mjModel* m, mjData* d)
{
	Generator g;
	const int number_of_inputs = m->nsensordata;
	const int number_of_outputs = m->nu;
	const int recurrent_inputs = n_outputs - number_of_outputs; //>=0
	//state in (number_of_inputs + recurrent_inputs) x 1

	ScalingLayer input_scaling(number_of_inputs + recurrent_inputs, 1);
	//input_scaling.getScaling().array() /= 2.0;
	const ScalarType scale_touch = 1.0 / 1000.0;
	input_scaling(0) = scale_touch;
	input_scaling(1) = scale_touch;
	input_scaling(2) = scale_touch;
	input_scaling(3) = scale_touch;

	//Copy input data
	for (int i = 0; i < number_of_inputs; i++)
		state(i) = d->sensordata[i];

	int k = number_of_outputs;
	for (int i = number_of_inputs; i < (number_of_inputs + recurrent_inputs); i++) {
		state(i) = controller->output()(k);
		k++;
	}

	input_scaling.input(state);
	state = input_scaling.output();

	controller->input(state);
	//controller->input(input_scaling.output());
	const MatrixType& output = controller->output();

	//Copy output data
	for (int i = 0; i < number_of_outputs; i++)
		d->ctrl[i] = output(i) + g.generate_normal<ScalarType>(0, 0.001);
}

// main function
void main()
{
	RandomEngineFactory::initialize();
	// activate software
	int activate_result = mj_activate("mjkey.txt");
	if (activate_result == 0) {
		std::cout << "Add mjkey.txt to mjenvironment/" << std::endl;
		std::cin.get();
		return;
	}

	//load and compile model
	loadmodel(model_name);

	// init GLFW
	if (!glfwInit())
		mju_error("Could not initialize GLFW");

	// create window, make OpenGL context current, request v-sync
	GLFWwindow* window = glfwCreateWindow(1920, 1080, "Demo", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);

	glfwSetKeyCallback(window, keyboard);
	glfwSetCursorPosCallback(window, mouse_move);
	glfwSetMouseButtonCallback(window, mouse_button);
	glfwSetScrollCallback(window, scroll);

	// initialize visualization data structures
	mjv_defaultCamera(&cam);
	mjv_defaultOption(&opt);
	mjr_defaultContext(&con);


	mjv_makeScene(&scn, 1000);                   // space for 1000 objects
	mjr_makeContext(m, &con, mjFONTSCALE_100);   // model-specific context
												 // install GLFW mouse and keyboard callbacks
												 //Create controller and hook callback to mj_step.
	createNeuralController();

	//sensors init
	sensorinit();

	// run main loop, target real-time simulation and 60 fps rendering
	while (!glfwWindowShouldClose(window))
	{
		// advance interactive simulation for 1/60 sec
		//  Assuming MuJoCo can simulate faster than real-time, which it usually can,
		//  this loop will finish on time for the next frame to be rendered at 60 fps.
		//  Otherwise add a cpu timer and exit this loop when it is time to render.
		mjtNum simstart = d->time;
		while (d->time - simstart < 1.0 / 60.0) {
			mj_step(m, d);
			test_controller(m, d);
		}

		// get size of active renderbuffer
		// get current framebuffer rectangle
		mjrRect rect = { 0, 0, 0, 0 };
		mjrRect viewport = mjr_maxViewport(&con);
		glfwGetFramebufferSize(window, &rect.width, &rect.height);

		// center and scale view
		cam.lookat[0] = d->qpos[0]+1;
		cam.lookat[1] = 0;
		cam.lookat[2] = d->qpos[1];
		cam.distance = 1.0 * m->stat.extent;

		//// update scene and render
		//glfwMakeContextCurrent(window);

		mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
		mjr_render(viewport, &scn, &con);

		glfwGetFramebufferSize(window, &rect.width, &rect.height);
		mjrRect smallrect = rect;

		sensorupdate();
		sensorshow(smallrect);

		// add time stamp in upper-left corner
		char stamp[50];
		sprintf(stamp, "Time = %.2f [s], Speed_x = %.2f [m/s]", d->time, d->qvel[0]);
		mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport, stamp, NULL, &con);

		// swap OpenGL buffers (blocking call due to v-sync)
		glfwSwapBuffers(window);

		// process pending GUI events, call GLFW callbacks
		glfwPollEvents();
	}
	destroyNeuralController();

	// close GLFW, free visualization storage
	glfwTerminate();
	mjv_freeScene(&scn);
	mjr_freeContext(&con);

	// free MuJoCo model and data, deactivate
	mj_deleteData(d);
	mj_deleteModel(m);

	mj_deactivate();
}

