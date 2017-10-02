#pragma once
#include "mujoco.h"
#include <iostream>
#include <atomic>
#include <stdexcept>
#include <stdlib.h> //conversions and shit

#include "NeuralNet.h"


class mjAgent 
{
public:
	mjAgent() : 
		m_model(nullptr), 
		m_data(nullptr), 
		m_controller(nullptr) {

	}
	mjAgent(const mjAgent& copy_this) = delete;
	mjAgent(mjAgent&& move_this) {
		m_model = move_this.m_model;
		m_data = move_this.m_data;
		m_controller = move_this.m_controller;

		move_this.m_model = nullptr;
		move_this.m_data = nullptr;
		move_this.m_controller = nullptr;
	}

	~mjAgent() {
		if(m_data)
			mj_deleteData(m_data);
		m_data = nullptr;
	}

	void setup(const mjModel const* model, int index = 0, int rows = 10, int cols = 10) {
		if (!model)
			throw std::runtime_error("Null model ptr.");
		m_model = model;

		if (m_data)
			mj_deleteData(m_data);

		int col = index % rows;
		int row = index / cols;

		model->qpos0[0] = 2 * col - cols; //Center rectangle
		model->qpos0[1] = 2 * row - rows;

		m_data = mj_makeData(m_model);
		mj_forward(m_model, m_data);
	}
	void reset() {
		mj_resetData(m_model, m_data);
		mj_forward(m_model, m_data);
	}

	void simulate() {
		mjtNum simstart = m_data->time;
		while (m_data->time - simstart < 1.0 / 240.0) {
			mj_step1(m_model, m_data);
			agentControll();
			mj_step2(m_model, m_data);
		}
	}
	
	void setController(NeuralNet* net) {
		m_controller = net;
	}

	mjData* getData() const { return m_data; }
protected:
	void agentControll() {
		if (m_controller) {
			//Setup states
			const int number_of_inputs = m_model->nsensordata;
			MatrixType input(number_of_inputs, 1);
			
			//Copy input data
			for (int i = 0; i < number_of_inputs; i++)
				input(i) = m_data->sensordata[i];

			m_controller->input(input);
			const MatrixType& output = m_controller->output();

			//Copy output data
			for (int i = 0; i < m_model->nu; i++)
				m_data->ctrl[i] = output(i);
		}
	}
	
private:
	// MuJoCo data structures
	mjModel const* m_model;           // MuJoCo model
	mjData* m_data;                   // MuJoCo data
	NeuralNet* m_controller;
};
