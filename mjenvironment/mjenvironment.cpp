#include "stdafx.h"
#include "FeedForwardNeuralNet.h"
#include "LayeredTopology.h"


#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include <iostream>
#include <memory>


#include <stdio.h>      /* printf, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */



#ifdef _DEBUG
#pragma comment(lib, "../x64/Debug/neuralnet.lib")
#else
#pragma comment(lib, "../x64/Release/neuralnet.lib")
#endif // _DEBUG

const int n_inputs = 5;
FeedForwardNeuralNet* ffnn = nullptr;
FeedForwardNeuralNet* feedbacknn = nullptr;
MatrixType insignal(n_inputs, 1);


// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjModel* m2 = NULL;
mjData* d = NULL;                   // MuJoCo data
mjData* d2 = NULL;
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con,con2;                     // custom GPU context

mjvFigure figsensor; //sensor figure

									// mouse interaction
bool controller_on = false;
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;
int sign = 1;

void createNeuralController() {
	std::vector<int> layerSizes = { n_inputs,		   40, 4000, 400, 40, 21 };
	std::vector<int> layerTypes = { Layer::inputLayer, 0,  0,  0,  0,  0 };
	LayeredTopology* top  = new LayeredTopology(layerSizes, layerTypes);

	ffnn = new FeedForwardNeuralNet(top); //memory is managed by network
	ffnn->initializeRandomWeights();

	std::vector<int> layerSizesfeedback = { 21, 40, 40, 40, 40, 3 };
	LayeredTopology* feedbacktop = new LayeredTopology(layerSizesfeedback, layerTypes);

	feedbacknn = new FeedForwardNeuralNet(feedbacktop);
	feedbacknn->initializeRandomWeights();
}

void destroyNeuralController() {
	if (ffnn)
		delete ffnn;
	ffnn = nullptr;
	if (feedbacknn)
		delete feedbacknn;
	feedbacknn = nullptr;
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
	if (!mnew)
	{
		printf("%s\n", error);
		return;
	}

	// delete old model, assign new
	mj_deleteData(d);
	mj_deleteModel(m);
	m = mnew;
	d = mj_makeData(m);
	mj_forward(m, d);

	mj_saveModel(m, "asd.mjb", NULL, 0);
	//mj_deleteModel(m2);
	//mj_deleteData(d2);
	//d2 = 0;
	//m2 = 0;
	//m2 = mj_loadXML("humanoid100.xml", 0, error, 1000);

	//if (!mnew) {
	//	printf("%s\n", error);
	//	return;
	//}

	//d2 = mj_makeData(m2);
	//mj_forward(m2, d2);
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
			figsensor.linedata[lineid][2 * p + 4 * i + 3] = (float)(d->sensordata[adr + i] / cutoff);
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
		std::cout << "1: " << d->sensordata[0] << std::endl;
		std::cout << "2: " << d->sensordata[1] << std::endl;
		std::cout << "acc x: " << d->sensordata[2] << std::endl;
		std::cout << "acc y: " << d->sensordata[3] << std::endl;
		std::cout << "acc z: " << d->sensordata[4] << std::endl;

		std::cout << "gyro x: " << d->sensordata[5] << std::endl;
		std::cout << "gyro y: " << d->sensordata[6] << std::endl;
		std::cout << "gyro z: " << d->sensordata[7] << std::endl;
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_R)
	{
		// load and compile model
		loadmodel("humanoid.xml");
	}

	if (act == GLFW_PRESS && key == GLFW_KEY_W)
	{
		insignal(0, 0) = 1;
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_S)
	{
		insignal(0, 0) = -1;
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_A)
	{
		insignal(1, 0) = 1;
	}
	if (act == GLFW_PRESS && key == GLFW_KEY_D)
	{
		insignal(1, 0) = -1;
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
	if (controller_on) {
		feedbacknn->input(ffnn->output());
		insignal.block(2, 0, 3, 1) = feedbacknn->output();
		ffnn->input(insignal);
		const MatrixType& outsignal = ffnn->output();
		//std::cout << outsignal;
		for (int i = 0; i < m->nu; i++) {
			*(d->act + i) = outsignal(i, 0);
		}
	}
	else {
		for (int i = 0; i < m->nu; i++) {
			*(d->act + i) = 0;
		}
	}
}


// main function
int main(int argc, const char** argv)
{
	srand(time(NULL));
	// activate software
	int activate_result = mj_activate("mjkey.txt");
	if (activate_result == 0) {
		std::cout << "Add mjkey.txt to mjenvironment/" << std::endl;
		std::cin.get();
		return 1;
	}

	//load and compile model
	loadmodel("humanoid.xml");

	// init GLFW
	if (!glfwInit())
		mju_error("Could not initialize GLFW");

	// create window, make OpenGL context current, request v-sync
	GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
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
	mjcb_control = test_controller;

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
		while (d->time - simstart < 1.0 / 60.0)
			mj_step(m, d);

		// get framebuffer viewport
		mjrRect viewport = { 0, 0, 0, 0 };
		glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

		// update scene and render
		mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
		mjr_render(viewport, &scn, &con);
		
		// get current framebuffer rectangle
		mjrRect rect = { 0, 0, 0, 0 };
		glfwGetFramebufferSize(window, &rect.width, &rect.height);
		mjrRect smallrect = rect;

		sensorupdate();
		sensorshow(smallrect);

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

	return 1;
}

