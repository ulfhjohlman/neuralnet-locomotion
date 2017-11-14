#pragma once
#include <iostream>


#include "mjAgent.h"
#include "NeuralNet.h"
#include "AgentScheduler.h"
#include "ThreadsafeQueue.h"
#include "ThreadPoolv2.h"
#include "utilityfunctions.h"

#include "glfw3.h"
#include "mujoco.h"

#include <future>
#include <mutex>


//Just work in progress
class mjEnvironment
{
public:
	mjEnvironment() = delete;
	mjEnvironment(int number_of_agents, int parallel_simulations, const char* model, const char* environment);
	~mjEnvironment();

	void setObjective(std::function<double(mjModel const *, mjData*)> objective) {
		m_objective = objective;
	}
	void setTerminateCondition(std::function<bool(mjModel const *, mjData*)> condition) {
		m_condition = condition;
	}
	void setScalingLayer(const ScalingLayer& scale_input) {
		m_agent_receptor_scaling = scale_input;
	}

	void evaluateController(NeuralNet* controller, size_t index) {
		std::lock_guard<std::mutex> lk(m_mutex);

		if (!controller)
			throw std::runtime_error("Null controller ptr.");

		if (m_agents.empty()) {
			auto new_agent = makeNewAgent();
			m_agents.push_back(std::move(new_agent));
		}
		
		auto agent = std::move(m_agents.back());
		m_agents.pop_back();
		agent->reset();

		agent->setController(controller);
		agent->setIdentifier(index);
		agent->setObjective(m_objective);
		agent->setTerminateCondition(m_condition);
		agent->setScalingLayer(m_agent_receptor_scaling);

		m_schedule.schedule(std::move(agent));
	}

	std::unique_ptr<mjAgent> makeNewAgent()
	{
		std::unique_ptr<mjAgent> pAgent = std::make_unique<mjAgent>();
		pAgent->setup(m_model, 0, 1, 1);
		m_number_of_agents++;
		return pAgent;
	}

	void simulate() {
		size_t agents_to_run = m_schedule.size();

		auto simulation = [this](int i) {
			auto agent = m_schedule.next();
			agent->simulate();
			if (agent->done()) {
				m_pending_simulation_result.push(agent->getFitness());
				m_mutex.lock();
				m_agents.push_back(std::move(agent)); //return agent to pool
				m_mutex.unlock();
				m_schedule.fill();
			}
			else {
				m_schedule.returnToSchedule(std::move(agent));
			}
		};

		parallel_for(0, (int)agents_to_run, 1, simulation);
	}
	bool all_done() {
		bool all_returned = m_agents.size() == m_number_of_agents;
		return all_returned;
		// std::all_of(m_agents.begin(), m_agents.end(), [](std::unique_ptr<mjAgent>& a) { return a->done(); })
	}

	bool tryGetResult( std::pair<size_t, double>& result ) {
		bool has_result = m_pending_simulation_result.try_pop(result);
		return has_result;
	}

	void render(const mjrRect& viewport) {
		if (!m_schedule.size())
			return;

		//Make scene
		mjv_updateScene(m_env, m_env_data, &opt, &pert, &cam, mjCAT_ALL, &scn);

		//Add running simulations to scene
		size_t agents_to_render = m_schedule.size();
		for (int i = 0; i < agents_to_render; i++) {
			auto agent = m_schedule.next();
			mjv_addGeoms(m_model, agent->getData(), &opt, &pert, mjCAT_ALL, &scn);
			m_schedule.returnToSchedule(std::move(agent));
		}
		
		//render
		mjr_render(viewport, &scn, &con);
	}

	void render(const mjrRect& viewport, const mjModel* model, mjData* data) {
		//Make scene
		mjv_updateScene(m_env, m_env_data, &opt, &pert, &cam, mjCAT_ALL, &scn);

		//Add agent data structure to visuals
		mjv_addGeoms(model, data, &opt, &pert, mjCAT_ALL, &scn);

		//render
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
	std::mutex m_mutex;
	std::vector< std::unique_ptr<mjAgent> > m_agents;
	AgentScheduler<mjAgent> m_schedule;

	ThreadsafeQueue<std::pair<size_t, double>> m_pending_simulation_result; //pair < index, fitness >
	int m_number_of_agents;

	std::function<double(mjModel const *, mjData*)> m_objective;
	std::function<bool(mjModel const *, mjData*)> m_condition;
	ScalingLayer m_agent_receptor_scaling;

private: //methods
	
private: //Class members
	static bool initializeMujoco();
	static std::atomic_int mj_instances;
};


mjEnvironment::mjEnvironment(
	int number_of_agents, 
	int parallel_simulations = 128, 
	const char* model = "invdoublependulum.xml", 
	const char* environment = "environment.xml") :
	m_objective(nullptr),
	m_condition(nullptr),
	m_schedule(parallel_simulations),
	m_number_of_agents(0)
{
	bool initOk = initializeMujoco();
	if (!initOk)
		throw std::runtime_error("Unable to activate Mujoco.");
	
	reload:
	try {
		//m_model = loadMujocoModel("invdoublependulum.xml");
		m_model = loadMujocoModel(model);
		m_env = loadMujocoModel(environment);
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
	std::cout << "Generating " << number_of_agents << " agents... ";
	for (int i = 0; i < number_of_agents; i++) {
		auto pAgent = this->makeNewAgent();
		m_agents.push_back(std::move(pAgent));
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