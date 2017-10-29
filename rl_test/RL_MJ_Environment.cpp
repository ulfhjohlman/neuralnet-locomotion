#pragma once
#include "mjenvironment.h"

class RL_MJ_Environment(): Environment {
	public:
        RL_MJ_Environment(){
            if (!init()){
                throw runtime_error("Unable to init() RL_MJ_environemnt!\n");
            }
            nsensors = environment->model->nsensordata;
        	nctrls	 = environment->model->nu;
        	agent.setup(model);

            state.resize(nsensors);
	}


    virtual void step(const std::vector<ScalarType>& actions)
    {
        double simstep = 0.003;

        // agent.simulate(1);
        mj_step1(m_model, m_data);
        agentControll();
        mj_step2(m_model, m_data);

        m_time_simulated += simstep*steps;

		agent.getState();
    }
    virtual ScalarType getReward()
    {
        return m_data->site_xpos[2] + m_data->site_xpos[0] - std::abs(m_data->site_xpos[1]);
    }
    virtual const std::vector<ScalarType>& getState()
    {
        //Copy input data
        for (int i = 0; i < nsensors; i++){
            state[i]] = m_data->sensordata[i];
        }
        return state;

    }
    virtual void reset()
    {
        m_model->qpos0[0] = 2 * m_col - m_cols; //Center rectangle
        m_model->qpos0[1] = 2 * m_row - m_rows;
        mj_resetData(m_model, m_data);
        mj_forward(m_model, m_data);
        m_integral = 0;
        m_done = false;
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
    mjData* m_data;
    mjAgent agent;
    mjModel* model = loadMujocoModel("humanoid.xml");
    std::vector<ScalarType> state;

    int nsensors;
    int nctrls;
    bool b_render = true;
    bool button_left = false;
    bool button_middle = false;
    bool button_right = false;
    double lastx = 0;
    double lasty = 0;
    double m_time_simulated = 0;

    GLFWwindow* window = nullptr;
    bool init();
    void setup(const mjModel const* model, int index = 0, int rows = 1, int cols = 1);
}



int main() {

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
RL_MJ_Environment::setup(const mjModel const* model, int index = 0, int rows = 1, int cols = 1) {
    if (!model)
        throw std::runtime_error("Null model ptr.");
    m_model = model;

    if (m_data) {
        mj_deleteData(m_data);
        m_data = nullptr;
    }
    m_data = mj_makeData(m_model);

    //Position for visual simulation
    this->setGridLocation(index, rows, cols);

    this->reset();
}

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

	glfwSetKeyCallback(window, keyboard);
	glfwSetCursorPosCallback(window, mouse_move);
	glfwSetMouseButtonCallback(window, mouse_button);
	glfwSetScrollCallback(window, scroll);

	return true;
}
