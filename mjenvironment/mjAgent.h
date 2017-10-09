#pragma once
#include "mujoco.h"
#include <iostream>
#include <atomic>
#include <stdexcept>
#include <stdlib.h> //conversions and shit
#include <functional>
#include <future>
#include <tuple>

#include "NeuralNet.h"


class mjAgent 
{
public:
	mjAgent() :
		m_model(nullptr),
		m_data(nullptr),
		m_controller(nullptr),
		m_integral(0.0),
		m_done(false),
		m_time_simulated(0.0),
		m_objective(nullptr),
		m_row(0),
		m_col(0),
		m_rows(0),
		m_cols(0),
		m_identifier(-1)
	{
	}
	mjAgent(const mjAgent& copy_this) = delete;
	mjAgent(mjAgent&& move_this) {
		m_model = move_this.m_model;
		m_data = move_this.m_data;
		m_controller = move_this.m_controller;

		move_this.m_model = nullptr;
		move_this.m_data = nullptr;
		move_this.m_controller = nullptr;

		m_integral = move_this.m_integral;
		m_done = move_this.m_done;
		m_time_simulated = move_this.m_time_simulated;
		m_objective = move_this.m_objective;

		m_row = move_this.m_row;
		m_rows = move_this.m_rows;
		m_col = move_this.m_col;
		m_cols = move_this.m_cols;

		m_identifier = move_this.m_identifier;
	}

	~mjAgent() {
		if(m_data)
			mj_deleteData(m_data);
		m_data = nullptr;
	}

	void setGridLocation(int index, int rows, int cols) {
		m_rows = rows;
		m_cols = cols;

		m_col = index % rows;
		m_row = index / cols;
	}

	void setup(const mjModel const* model, int index = 0, int rows = 1, int cols = 1) {
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

	void reset() {
		m_model->qpos0[0] = 2 * m_col - m_cols; //Center rectangle
		m_model->qpos0[1] = 2 * m_row - m_rows;

		mj_resetData(m_model, m_data);
		mj_forward(m_model, m_data);

		m_integral = 0;
		m_done = false;
		m_time_simulated = 0;
	}

	void simulate( int steps = 40 ) {
		double simstep = 0.003;
		
		if(!m_done)
			for (int i = 0; i < steps; i++) {
				mj_step1(m_model, m_data);
				agentControll();
				mj_step2(m_model, m_data);

				if (m_objective) {
					m_integral += simstep * m_objective(m_model, m_data);
				}
			}
			
			m_time_simulated += simstep*steps;
			if (m_time_simulated > 7.0) {
				m_done = true;
			}
	}
	
	bool done() {
		return m_done;
	}

	std::pair<size_t, double> getFitness() {
		return std::make_pair(m_identifier, m_integral);
	}

	void setController(NeuralNet* net) {
		m_controller = net;
	}
	void setObjective(std::function<double(mjModel const *, mjData*)> objective) {
		m_objective = objective;
	}
	void setIdentifier(size_t identifier) {
		m_identifier = identifier;
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

	std::function<double( mjModel const *, mjData*)> m_objective;
	std::promise<double> m_fitness_measure;

	bool m_done;
	double m_time_simulated;
	double m_integral;

	int m_col, m_cols;
	int m_row, m_rows;

	size_t m_identifier; //Index in population for concurrent retrieval
};
