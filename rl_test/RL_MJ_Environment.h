#pragma once
#include "mjEnvironment.h"
#include "Environment.h"
#include "mujoco.h"
#include "mjAgent.h"
#include "config.h"


bool b_render = true;

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



class RL_MJ_Environment: public Environment {
	public:

    virtual void step(const std::vector<ScalarType>& actions)
    {
        #ifdef _DEBUG
            if(actions.size() != nctrls)
            {
                throw std::runtime_error("Controler actions.size() != nctrls\n");
            }
        #endif
        double simstep = 0.003;

        // agent.simulate(1);
		double pos_before = data->qpos[0, 0]; 
		double actionSqrSum = 0;
		for (int i = 0; i < frameskip; i++)
		{
			mj_step1(model, data);
			if (i == 0) {
				for (int i = 0; i < model->nu; i++)
				{
					data->ctrl[i] = actions[i];
					actionSqrSum += actions[i] * actions[i];
				}
			}
			mj_step2(model, data);
			render();
		}
		double pos_after = data->qpos[0, 0];
		velocity = (pos_after - pos_before) / (simstep * frameskip);
        m_time_simulated += simstep*frameskip;
		++step_nr;


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

    int nsensors;
    int nctrls;
	double cntrlCostCoef = 1e-4;
	//double cntrlCostCoef = 100;
	int frameskip = 10;
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
			//model = environment->loadMujocoModel("humanoid.xml");

			agent.setup(model);
			data = agent.getData();
			//state.resize(nsensors);
			state.resize(model->nq + model->nv);
			mj_forward(model, data);

			glfwSetKeyCallback(window, keyboard);

			mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
			//environment->cam.type = mjCAMERA_TRACKING;
			//environment->cam.trackbodyid = 0;
	}
	virtual double getReward() {
		return velocity - 1e-3*actionSqrSum + 1;
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
		return false;
	}
	AntEnv() {
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

		glfwSetKeyCallback(window, keyboard);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
	}
	virtual double getReward() {
		return velocity - 1e-4*actionSqrSum + 0.02;
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

		glfwSetKeyCallback(window, keyboard);

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
		return (data->site_xpos[5]<0);
	}
	InvDoublePendEnv() {
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

		glfwSetKeyCallback(window, keyboard);

		mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
	}
	virtual double getReward() {
		return data->site_xpos[5];  //invdoublepenulum;
	}

};
