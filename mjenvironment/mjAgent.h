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
#include "Generator.h"
#include "ScalingLayer.h"

int g_simulation_steps = 10;
float g_minimum_kill_height = 0.26f;
float g_max_simulation_time = 1.0f;


class mjAgent 
{
public:
	mjAgent() :
		//data structures
		m_model(nullptr),
		m_data(nullptr),

		//neural nets
		m_controller(nullptr),
		m_receptors(),

		//function pointers
		m_objective(nullptr),
		m_condition(nullptr),

		m_reward(0.0),
		m_done(false),
		m_has_scaling_layer(false),
		m_time_simulated(0.0),
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
		m_receptors = move_this.m_receptors; //not a pointer

		move_this.m_model = nullptr;
		move_this.m_data = nullptr;
		move_this.m_controller = nullptr;

		m_reward = move_this.m_reward;
		m_time_simulated = move_this.m_time_simulated;

		m_done = move_this.m_done;
		m_has_scaling_layer = move_this.m_has_scaling_layer;

		
		m_objective = move_this.m_objective;
		m_condition = move_this.m_condition;

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

		m_col = index % m_cols;
		m_row = index / m_rows;
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

		m_reward = 0;
		m_done = false;
		m_time_simulated = 0;
	}

	void simulate( int steps = 10 ) {
		double simstep = 0.001;
		
		steps = g_simulation_steps;
		if(!m_done)
			for (int i = 0; i < steps; i++) {
				mj_step(m_model, m_data);
				this->controll();

				if (m_objective) {
					m_reward += simstep * m_objective(m_model, m_data);
					m_reward += - 0.000001 * m_controller->output().squaredNorm();
				}
				
			}
			m_time_simulated += simstep*steps;
			if (m_time_simulated > g_max_simulation_time || m_data->site_xpos[5] < g_minimum_kill_height) {
				m_done = true;
			}
			//|| m_data->site_xpos[5] < g_minimum_kill_height

	}
	
	bool done() {
		return m_done;
	}

	std::pair<size_t, double> getFitness() {
		return std::make_pair(m_identifier, m_reward);
	}

	void setController(NeuralNet* net) {
		m_controller = net;
	}
	void setObjective(std::function<double(mjModel const *, mjData*)> objective) {
		m_objective = objective;
	}
	void setTerminateCondition(std::function<bool(mjModel const *, mjData*)> condition) {
		m_condition = condition;
	}
	void setScalingLayer(const ScalingLayer& scale_input) {
		m_has_scaling_layer = true;
		m_receptors = scale_input;
	}
	void setIdentifier(size_t identifier) {
		m_identifier = identifier;
	}

	mjData* getData() const { return m_data; }
protected:
	void controll() {
		Generator g;
		if (m_controller) {
			//Setup states
			const int number_of_inputs  = m_model->nsensordata;
			const int number_of_outputs = m_model->nu;
			const int recurrent_inputs  = 16;
			MatrixType input(number_of_inputs + recurrent_inputs, 1);

			//Copy input data
			for (int i = 0; i < number_of_inputs; i++)
				input(i) = m_data->sensordata[i];

			int k = number_of_outputs;
			for (int i = number_of_inputs; i < (number_of_inputs + recurrent_inputs); i++) {
				input(i) = m_controller->output()(k);
				k++;
			}

			if (m_has_scaling_layer) {
				m_receptors.input(input);
				m_controller->input(m_receptors.output());
			}
			else {
				m_controller->input(input);
			}

			//Copy output data
			for (int i = 0; i < number_of_outputs; i++)
				m_data->ctrl[i] = -1;
		}
	}
	
private:
	// MuJoCo data structures
	mjModel const* m_model;           // MuJoCo model
	mjData* m_data;                   // MuJoCo data
	NeuralNet* m_controller;
	ScalingLayer m_receptors;

	std::function<double( mjModel const *, mjData*)> m_objective;
	std::function<bool(mjModel const *, mjData*)> m_condition;

	bool m_done;
	bool m_has_scaling_layer;
	double m_time_simulated;
	double m_reward;

	int m_col, m_cols;
	int m_row, m_rows;

	size_t m_identifier; //Index in population for concurrent retrieval
};
/*m_data->sensordata[0] /= 500.0;
m_data->sensordata[1] /= 500.0;
m_data->sensordata[2] /= 500.0;
m_data->sensordata[3] /= 500.0;

m_data->sensordata[5] /= 10.0;
m_data->sensordata[6] /= 10.0;
m_data->sensordata[7] /= 10.0;

m_data->sensordata[11] /= 10.0;
m_data->sensordata[12] /= 10.0;
m_data->sensordata[13] /= 10.0;*/

//m_data->sensordata[4] /= 500.0;
//m_data->sensordata[5] /= 500.0;
//
////scale acc
//m_data->sensordata[6] /= 10.0;
//m_data->sensordata[7] /= 10.0;
//m_data->sensordata[8] /= 10.0;

//Scale gyro
/*m_data->sensordata[9] /= 3.0;
m_data->sensordata[10] /= 3.0;
m_data->sensordata[11] /= 3.0;*/
