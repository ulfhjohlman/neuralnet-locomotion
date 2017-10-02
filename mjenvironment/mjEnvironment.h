#pragma once
#include <iostream>
#include "glfw3.h"

#include "mjAgent.h"
#include "NeuralNet.h"

//Just work in progress
class mjEnvironment
{
public:
	mjEnvironment() = delete;
	mjEnvironment(int number_of_agents);
	~mjEnvironment();

	void evaluateController(NeuralNet* controller) {
		if (!controller)
			throw std::runtime_error("Null controller ptr.");
		m_agents[it].setController(controller);
		it++;
		if (it >= m_agents.size())
			it = 0;
	}
	void render(mjrRect& viewport) {
		if (m_agents.empty())
			return;
		
#pragma omp parallel for
			for (int i = 0; i < m_agents.size(); i++)
				m_agents[i].simulate();

		mjv_updateScene(m_env, m_env_data, &opt, &pert, &cam, mjCAT_ALL, &scn);
		for (int i = 0; i < m_agents.size(); i++) {
			mjv_addGeoms(m_model, m_agents[i].getData(), &opt, &pert, mjCAT_ALL, &scn);
		}

		mjr_render(viewport, &scn, &con);
	}

	mjModel* loadMujocoModel(const char* filename);

	mjModel* m_model;
	mjModel* m_env;
	mjData* m_env_data;

	mjvCamera cam;                      // abstract camera
	mjvOption opt;                      // visualization options
	mjvScene scn;                       // abstract scene
	mjrContext con;                     // custom GPU context
	mjvPerturb pert;
private:
	int it;
	std::vector< mjAgent > m_agents;

	
private: //Class members
	static bool initializeMujoco();
	static std::atomic_int mj_instances;
};


mjEnvironment::mjEnvironment(int number_of_agents) : it(0)
{
	bool initOk = initializeMujoco();
	if (!initOk)
		throw std::runtime_error("Unable to activate Mujoco.");
	
	reload:
	try {
		m_model = loadMujocoModel("humanoid.xml");
		m_env = loadMujocoModel("environment.xml");
	}
	catch(std::runtime_error e) {
		std::cerr << e.what() << std::endl;
		std::cerr << "Press any key to reload\n";
		std::cin.get();
		goto reload;
	}
	m_env_data = mj_makeData(m_env);
	mj_forward(m_env, m_env_data);

	// initialize visualization data structures
	std::cout << "Initializing mujoco visuals... ";
	mjv_defaultCamera(&cam);
	mjv_defaultOption(&opt);
	mjr_defaultContext(&con);
	mjv_defaultPerturb(&pert);

	mjv_makeScene(&scn, 100000);                   // space for 100 000 objects
	mjr_makeContext(m_model, &con, mjFONTSCALE_100);   // model-specific context
	std::cout << "done\n";

	m_agents.reserve(number_of_agents);
	std::cout << "generating " << number_of_agents << " agents... ";
	for (int i = 0; i < number_of_agents; i++) {
		mjAgent agent;
		agent.setup(m_model, i);
		m_agents.push_back(std::move(agent));
	}
	std::cout << " done.\n";
}

mjEnvironment::~mjEnvironment()
{
	if (m_model)
		mj_deleteModel(m_model);
	m_model = nullptr;

	if (m_env)
		mj_deleteModel(m_env);
	m_env = nullptr;
	if (m_env_data)
		mj_deleteData(m_env_data);
	m_env_data = nullptr;

	mjr_freeContext(&con);
	mjv_freeScene(&scn);

	int instances_before = mj_instances.fetch_sub(1);
	if (instances_before == 1) {
		mj_deactivate();
		std::cout << "Mujoco deactivated\n";
	}
}





std::atomic_int mjEnvironment::mj_instances = 0;
bool mjEnvironment::initializeMujoco()
{
	std::cout << "Initializing Mujoco... ";
	int instances_before = mj_instances.fetch_add(1);
	if (instances_before == 0) {
		int activate_result = mj_activate("mjkey.txt");
		if (activate_result == 0) {
			std::cout << " failed" << std::endl;
			return false;
		}
	}
	std::cout << "Successfully" << std::endl;
	return true; //successful
}


// load mjb or xml model
mjModel* mjEnvironment::loadMujocoModel(const char* filename)
{
	// make sure filename is given
	if (!filename)
		throw std::runtime_error("No filename to load.");

	// load and compile
	char error[1000] = "could not load binary model";
	mjModel* mnew = 0;
	if (strlen(filename) > 4 && !strcmp(filename + strlen(filename) - 4, ".mjb"))
		mnew = mj_loadModel(filename, 0);
	else
		mnew = mj_loadXML(filename, 0, error, 1000);
	if (!mnew)
		throw std::runtime_error(error);

	//mj_printModel(mnew, "model_file.txt");
	return mnew;
}