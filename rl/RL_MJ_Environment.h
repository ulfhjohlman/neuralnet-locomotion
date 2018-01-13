#pragma once
#include "mjEnvironment.h"
#include "Environment.h"
#include "mujoco.h"
#include "mjAgent.h"
#include "config.h"


bool b_render = true;
bool button_left = false;
bool button_middle = false;
bool button_right = false;
double lastx = 0;
double lasty = 0;
mjEnvironment* global_environment = nullptr;


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
	if (global_environment)
		mjv_moveCamera(global_environment->m_model, action, dx / height, dy / height, &global_environment->scn, &global_environment->cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
	// emulate vertical mouse motion = 5% of window height
	if (global_environment)
		mjv_moveCamera(global_environment->m_model, mjMOUSE_ZOOM, 0, -0.05*yoffset, &global_environment->scn, &global_environment->cam);
}




class RL_MJ_Environment: public Environment {
	public:

		virtual void step(const std::vector<ScalarType>& actions)
		{
#ifdef _DEBUG
			if (actions.size() != nctrls)
			{
				throw std::runtime_error("Controler actions.size() != nctrls\n");
			}
#endif


			// agent.simulate(1);
			double pos_before = data->subtree_com[0];
			double actionSqrSum = 0;
			for (int i = 0; i < frameskip; i++)
			{
				//mj_step(model, data)
				if (i == 0) {
					//if(m_time_simulated==0)std::cout << "Ac: [";
					for (int j = 0; j < model->nu; j++)
					{
						//if (m_time_simulated == 0)std::cout << " " << actions[j] << "  ";
						data->ctrl[j] = actions[j];
						actionSqrSum += actions[j] * actions[j];
					}
					//if (m_time_simulated == 0)std::cout << "]\n";
				}
				mj_step(model, data);
				render();
			}
			extForceSqrSum = calcExtForceSqrSum();
			double pos_after = data->subtree_com[0];
			//velocity = (pos_after - pos_before) / (simstep * frameskip);
			velocity = data->subtree_linvel[0];
        m_time_simulated += simstep*frameskip;
		++step_nr;


    }
	double calcExtForceSqrSum() 
	{
		double x = 0;
		for (int i = 0; i < 6 * model->nbody; i++)
		{
			x += data->cfrc_ext[i]* data->cfrc_ext[i];
		}
		return x;
	}

	virtual double getReward() = 0;

    virtual const std::vector<ScalarType>& getState()
    {
        //Copy input data
        //for (int i = 0; i < nsensors; i++){
        //    state[i] = data->sensordata[i];
        //}
		for (int i = 0; i < model->nq; i++) {
			state[i] = data->qpos[i];
		}
		for (int i = 0; i < model->nv; i++) {
			state[model->nq+i] = data->qvel[i];
		}
        return state;

    }
    virtual void reset()
    {
        mj_resetData(model, data);
        mj_forward(model, data);
        m_time_simulated = 0;
		step_nr = 0;
		for (int i = 0; i < model->nq; i++) {
			data->qpos[i] += generator.generate_uniform(-0.005, 0.005);
		}
		for (int i = 0; i < model->nv; i++) {
			data->qvel[i] += generator.generate_uniform(-0.005, 0.005);;
		}
    }
    virtual int getActionSpaceDimensions()
    {
        return nctrls;
    }
    virtual int getStateSpaceDimensions()
    {
       // return nsensors;
		return (model->nq + model->nv);
    }
	virtual void set_frameskip(int new_skip)
	{
		frameskip = new_skip;
	}
protected:
	virtual void loadEnv() { std::cout << "bajs\n"; };

	void render()
	{
		if (!glfwWindowShouldClose(window))
		{
			//render here
			if (b_render) {
				// get framebuffer viewport
				mjrRect viewport = { 0, 0, 0, 0 };
				glfwGetFramebufferSize(window, &viewport.width, &viewport.height);



				environment->render(viewport, model, data);

				// swap OpenGL buffers (blocking call due to v-sync)
				glfwSwapBuffers(window);
			}

			// process pending GUI events, call GLFW callbacks
			glfwPollEvents();
		}
	}

    mjEnvironment* environment;
    mjData* data;
    mjAgent agent;
	mjModel* model;
    std::vector<ScalarType> state;
	Generator generator;
	double simstep= 0.003;

    int nsensors;
    int nctrls;
	
	//double cntrlCostCoef = 100;
	int frameskip = 1;
    bool button_left = false;
    bool button_middle = false;
    bool button_right = false;
	int m_col = 0;
	int m_cols = 0;
	int m_row = 0;
	int m_rows = 0;
    double lastx = 0;
    double lasty = 0;
    double m_time_simulated = 0;
	int step_nr = 0;
	double rew_quotient = 0;
	double actionSqrSum = 0;
	double extForceSqrSum = 0;
	double velocity = 0;

    GLFWwindow* window = nullptr;
	bool init();

};



bool RL_MJ_Environment::init() {
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


	return true;
}


/*
class HumanoidEnv2 : public RL_MJ_Environment
{
protected:

	virtual void loadEnv() {
		model = environment->loadMujocoModel("humanoid2.xml");

	}
public:
	virtual bool earlyAbort() {

		double height = data->qpos[2];

		//for (int i =0; i < 15 ; i++)
		//std::cout << "geom_xpos[" << i << "]: " << data->geom_xpos[i] << "\n";
		//return height < 1.0 || height > 2.0;
		return false;
	}

	HumanoidEnv2() {
		if (!init()) {
			throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
		}
		environment = new mjEnvironment(1, 1);
		loadEnv();
		nsensors = model->nsensordata;
		nctrls = model->nu;

		agent.setup(model);
		data = agent.getData();

		state.resize(getStateSpaceDimensions());
		mj_forward(model, data);

		global_environment = environment;
		glfwSetKeyCallback(window, keyboard);
		glfwSetCursorPosCallback(window, mouse_move);
		glfwSetMouseButtonCallback(window, mouse_button);
		glfwSetScrollCallback(window, scroll);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -3, -3, &environment->scn, &environment->cam);
		//environment->cam.type = mjCAMERA_TRACKING;
		//environment->cam.trackbodyid = 0;
	}
	virtual double getReward() {
		extForceSqrSum = fmin(extForceSqrSum, 10.0);
		return 0.25*velocity - 1e-1*actionSqrSum - 0.5*1e-6*extForceSqrSum + 5;
	}
	virtual const std::vector<ScalarType>& getState()
	{
		//messy concatenation :S
		for (int i = 0; i < model->nq; i++) {
			state[i] = data->qpos[i];
		}
		for (int i = 0; i < model->nv; i++) {
			state[model->nq + i] = data->qvel[i];
		}
		for (int i = 0; i < model->nbody * 10; i++) {
			state[model->nq + model->nv + i] = data->cinert[i];
		}
		for (int i = 0; i < model->nbody * 6; i++) {
			state[model->nq + model->nv + model->nbody * 10 + i] = data->cvel[i];
		}
		for (int i = 0; i < model->nv; i++) {
			state[model->nq + model->nv + model->nbody * 16 + i] = data->qfrc_actuator[i];
		}
		for (int i = 0; i < model->nbody * 6; i++) {
			state[model->nq + model->nv * 2 + model->nbody * 16 + i] = data->cfrc_ext[i];
		}
		return state;

	}
	virtual int getStateSpaceDimensions()
	{
		// return nsensors;
		return (model->nq + model->nv * 2 + model->nbody * 22);
	}
};
*/

class HumanoidEnv : public RL_MJ_Environment
{
protected:

	virtual void loadEnv() {
		model = environment->loadMujocoModel("humanoid.xml");
		
	}
public:
	virtual bool earlyAbort() {

		double height = data->qpos[2];

		//for (int i =0; i < 15 ; i++)
		//std::cout << "geom_xpos[" << i << "]: " << data->geom_xpos[i] << "\n";
		return height < 0.9 || height > 2.0;
		//return false;
	}

	HumanoidEnv() {
		if (!init()) {
			throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
		}
		environment = new mjEnvironment(1, 1);
		loadEnv();
		nsensors = model->nsensordata;
		nctrls = model->nu;
		simstep = 0.004;
		agent.setup(model);
		data = agent.getData();

		state.resize(getStateSpaceDimensions());
		mj_forward(model, data);

		global_environment = environment;
		glfwSetKeyCallback(window, keyboard);
		glfwSetCursorPosCallback(window, mouse_move);
		glfwSetMouseButtonCallback(window, mouse_button);
		glfwSetScrollCallback(window, scroll);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -3, -3, &environment->scn, &environment->cam);
		//environment->cam.type = mjCAMERA_TRACKING;
		//environment->cam.trackbodyid = 0;
	}
	virtual double getReward() {
		//extForceSqrSum = fmin(extForceSqrSum, 10.0);
		return velocity - 0.00002*actionSqrSum + simstep*0.1;
	}
	virtual const std::vector<ScalarType>& getState()
	{
		//messy concatenation :S
		for (int i = 0; i < nsensors; i++) {
			state[i] = data->sensordata[i];
		}
		return state;

	}
	virtual void step(const std::vector<ScalarType>& actions) {
		double pos_before = data->subtree_com[0];
		double actionSqrSum = 0;
		for (int i = 0; i < frameskip; i++)
		{
			//mj_step(model, data)
			if (i == 0) {
				//if(m_time_simulated==0)std::cout << "Ac: [";
				for (int j = 0; j < model->nu; j++)
				{
					//if (m_time_simulated == 0)std::cout << " " << actions[j] << "  ";
					data->ctrl[j] = data->ctrl[j]*0.1 + 0.9*actions[j];
					actionSqrSum += actions[j] * actions[j];
				}
				//if (m_time_simulated == 0)std::cout << "]\n";
			}
			mj_step(model, data);
			render();
		}
		extForceSqrSum = calcExtForceSqrSum();
		double pos_after = data->subtree_com[0];
		//velocity = (pos_after - pos_before) / (simstep * frameskip);
		velocity = data->subtree_linvel[0];
		m_time_simulated += simstep*frameskip;
		++step_nr;
	}
	virtual int getStateSpaceDimensions()
	{
		// return nsensors;
		//return (model->nq + model->nv*2 + model->nbody*22);
		return nsensors;
	}
};
class HopperEnv : public RL_MJ_Environment
{
protected:

	virtual void loadEnv() {
		model = environment->loadMujocoModel("hopper.xml");
	}
public:
	virtual bool earlyAbort() {

		double height = data->geom_xpos[5];
		
		//for (int i =0; i < 15 ; i++)
		//std::cout << "geom_xpos[" << i << "]: " << data->geom_xpos[i] << "\n";
		return height < 0.7;
	}

	HopperEnv(){
			if (!init()) {
				throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
			}
			environment = new mjEnvironment(1, 1);
			loadEnv();
			nsensors = model->nsensordata;
			nctrls = model->nu;

			agent.setup(model);
			data = agent.getData();
			//state.resize(nsensors);
			state.resize(model->nq + model->nv);
			mj_forward(model, data);

			global_environment = environment;
			glfwSetKeyCallback(window, keyboard);
			glfwSetCursorPosCallback(window, mouse_move);
			glfwSetMouseButtonCallback(window, mouse_button);
			glfwSetScrollCallback(window, scroll);

			mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
			//environment->cam.type = mjCAMERA_TRACKING;
			//environment->cam.trackbodyid = 0;
	}
	virtual double getReward() {
		return velocity - 1e-3*actionSqrSum + 1;
	}

};
class Walker2dEnv : public RL_MJ_Environment
{
protected:

	virtual void loadEnv() {
		model = environment->loadMujocoModel("walker2d.xml");
	}
public:
	virtual bool earlyAbort() {

		double height = data->geom_xpos[5];
		double ang = data->qpos[2];
		/*
		for (int i = 0; i < model->ngeom*3; i++)
			std::cout << "geom_xpos[" << i << "]: " << data->geom_xpos[i] << "\n";
		for (int i = 0; i < model->nq; i++)
			std::cout << "qpos[" << i << "]: " << data->qpos[i] << "\n";
		*/
		//return false;
		return !((height > 0.8 && height < 2) && (ang > -1 && ang < 1)) ;
	}

	Walker2dEnv() {
		if (!init()) {
			throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
		}
		simstep = 0.002;
		environment = new mjEnvironment(1, 1);
		loadEnv();
		nsensors = model->nsensordata;
		nctrls = model->nu;

		agent.setup(model);
		data = agent.getData();
		//state.resize(nsensors);
		state.resize(model->nq + model->nv);
		mj_forward(model, data);

		global_environment = environment;
		glfwSetKeyCallback(window, keyboard);
		glfwSetCursorPosCallback(window, mouse_move);
		glfwSetMouseButtonCallback(window, mouse_button);
		glfwSetScrollCallback(window, scroll);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);

	}
	virtual double getReward() {
		return velocity - 1e-5*actionSqrSum + 0.01;
	}

};
class AntEnv : public RL_MJ_Environment
{
protected:

	virtual void loadEnv() {
		model = environment->loadMujocoModel("ant.xml");
	}
public:
	virtual bool earlyAbort() {
		double height = data->geom_xpos[5];
		double ang = data->qpos[2];
		double limit_angle = 2; //radians
		return height < 0.25;
		//return false;
	}
	AntEnv() {
		if (!init()) {
			throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
		}
		environment = new mjEnvironment(1, 1);
		simstep = 0.01;
		loadEnv();
		nsensors = model->nsensordata;
		nctrls = model->nu;

		agent.setup(model);
		state.resize(model->nv + nsensors);
		data = agent.getData();
		mj_forward(model, data);

		global_environment = environment;
		glfwSetKeyCallback(window, keyboard);
		glfwSetCursorPosCallback(window, mouse_move);
		glfwSetMouseButtonCallback(window, mouse_button);
		glfwSetScrollCallback(window, scroll);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
	}
	virtual double getReward() {
		return velocity - 1e-4*actionSqrSum + 0.02;
	}

	virtual int getStateSpaceDimensions()
	{
		return (nsensors + model->nv);
	}

	virtual const std::vector<ScalarType>& getState()
	{
		for (int i = 0; i < nsensors; i++) {
			state[i] = data->sensordata[i];
		}
		for (int i = 0; i < model->nv; i++) {
			state[nsensors + i] = data->qvel[i];
		}
		return state;
	}
};

class SwimmerEnv : public RL_MJ_Environment
{
protected:

	virtual void loadEnv() {
		model = environment->loadMujocoModel("swimmer.xml");
	}
public:
	virtual bool earlyAbort() {
		return false; 
	}

	SwimmerEnv() {
		if (!init()) {
			throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
		}
		environment = new mjEnvironment(1, 1);
		loadEnv();
		nsensors = model->nsensordata;
		nctrls = model->nu;
		//model = environment->loadMujocoModel("humanoid.xml");

		agent.setup(model);
		data = agent.getData();
		//state.resize(nsensors);
		state.resize(model->nq + model->nv);
		mj_forward(model, data);

		global_environment = environment;
		glfwSetKeyCallback(window, keyboard);
		glfwSetCursorPosCallback(window, mouse_move);
		glfwSetMouseButtonCallback(window, mouse_button);
		glfwSetScrollCallback(window, scroll);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
	}
	virtual double getReward() {
		return velocity - 1e-4*actionSqrSum + 0.02;
	}
};

class InvDoublePendEnv : public RL_MJ_Environment
{
protected:

	virtual void loadEnv() {
		model = environment->loadMujocoModel("invdoublependulum.xml");
	}
public:
	virtual bool earlyAbort() {
		return (data->site_xpos[5]<1);
	}
	InvDoublePendEnv() {
		if (!init()) {
			throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
		}
		environment = new mjEnvironment(1, 1);
		loadEnv();
		nsensors = model->nsensordata;
		nctrls = model->nu;

		agent.setup(model);
		data = agent.getData();
		//state.resize(nsensors);
		state.resize(model->nq + model->nv);
		mj_forward(model, data);

		global_environment = environment;
		glfwSetKeyCallback(window, keyboard);
		glfwSetCursorPosCallback(window, mouse_move);
		glfwSetMouseButtonCallback(window, mouse_button);
		glfwSetScrollCallback(window, scroll);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
	}

	virtual double getReward() {
		return data->site_xpos[5];  //invdoublepenulum;
	}

};
