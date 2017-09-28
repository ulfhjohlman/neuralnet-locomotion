#pragma once
#include "mujoco.h"
#include <iostream>
#include <atomic>
#include <stdexcept>
#include <stdlib.h> //conversions and shit
#include <glfw3.h>

class MujocoWrapper 
{
public:
	MujocoWrapper() : m_model(nullptr), m_data(nullptr) {
		bool initOk = initializeMujoco();
		if (!initOk)
			throw std::runtime_error("Unable to activate Mujoco.");
	}
	~MujocoWrapper() {
		mj_deleteData(m_data);

		int instances_before = mj_instances.fetch_sub(1);
		if (instances_before == 1) //last one
			mj_deactivate();
	}

	void setup(mjModel* model) {
		if (!model)
			throw std::runtime_error("Null model ptr.");
		m_model = model;

		if (m_data)
			mj_deleteData(m_data);
		m_data = mj_makeData(m_model);
		mj_forward(m_model, m_data);
	}

	void simulatePhysics() {
		mjtNum simstart = m_data->time;
		while (m_data->time - simstart < 1.0 / 30.0)
			mj_step(m_model, m_data);
	}
	
	mjData* getData() const { return m_data; }

private:
	// MuJoCo data structures
	mjModel* m_model;                  // MuJoCo model
	mjData* m_data;                   // MuJoCo data

	static std::atomic_int mj_instances;
	static bool initializeMujoco();
};

// load mjb or xml model
mjModel* loadMujocoModel(const char* filename);