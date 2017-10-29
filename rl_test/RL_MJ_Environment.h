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
        RL_MJ_Environment(){

			if (!init()){
                throw std::runtime_error("Unable to init() RL_MJ_environemnt!\n");
            }
			environment = new mjEnvironment(1,1);
            nsensors = environment->m_model->nsensordata;
        	nctrls	 = environment->m_model->nu;
			model = environment->loadMujocoModel("humanoid.xml");
			agent.setup(model);
			data = agent.getData();
            state.resize(nsensors);
			mj_forward(model, data);

			glfwSetKeyCallback(window, keyboard);

			mjv_moveCamera(model, mjMOUSE_ZOOM, -1, -1, &environment->scn, &environment->cam);
	}


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
        mj_step1(model, data);
        for (int i = 0; i < model->nu; i++)
        {
            data->ctrl[i] = actions[i];
        }
        mj_step2(model, data);

        m_time_simulated += simstep;

		

		if(!glfwWindowShouldClose(window))
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
    virtual ScalarType getReward()
    {
        return data->site_xpos[2] + data->site_xpos[0] - std::abs(data->site_xpos[1]);
    }
    virtual const std::vector<ScalarType>& getState()
    {
        //Copy input data
        for (int i = 0; i < nsensors; i++){
            state[i] = data->sensordata[i];
        }
        return state;

    }
    virtual void reset()
    {
        model->qpos0[0] = 2 * m_col - m_cols; //Center rectangle
        model->qpos0[1] = 2 * m_row - m_rows;
        mj_resetData(model, data);
        mj_forward(model, data);

        m_time_simulated = 0;
    }
    virtual int getActionSpaceDimensions()
    {
        return nctrls;
    }
    virtual int getStateSpaceDimensions()
    {
        return nsensors;
    }
private:
    mjEnvironment* environment;
    mjData* data;
    mjAgent agent;
	mjModel* model;
    std::vector<ScalarType> state;

    int nsensors;
    int nctrls;
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
