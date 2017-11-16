#pragma once

#include "Individual.h"
#include "LayeredNeuralNet.h"
#include "CascadeTopology.h"
#include "CascadeNeuralNet.h"

#include "NeuralNetGenome.h"

class NeuralNetChromosome : public Individual<LayeredNeuralNet> //Add genome template
{
public:
	NeuralNetChromosome(int n_inputs, int n_ouputs) : m_controller(nullptr),
		m_number_of_inputs(n_inputs),
		m_number_of_outputs(n_ouputs) {
		createNeuralController();
	}
	virtual ~NeuralNetChromosome() {
		destroyNeuralController();
	}

	NeuralNetChromosome& operator=(const NeuralNetChromosome& copy_this) {
		//reallocate network? no need yet
		m_number_of_inputs = copy_this.m_number_of_inputs;
		m_number_of_outputs = copy_this.m_number_of_outputs;

		return *this;
	}

	virtual LayeredNeuralNet * decode() { return m_controller; }

	virtual void save(const char* path) { }
	virtual void load(const char* path) { }

protected:


	
private:
	LayeredNeuralNet * m_controller;
	int m_number_of_inputs;
	int m_number_of_outputs;

private:
	void createNeuralController() {
		/*std::vector<int> layerSizes = { m_number_of_inputs, 64, 64, 64, 64, 64, m_number_of_outputs };
		std::vector<int> layerTypes = { Layer::inputLayer, 1, 1, 1, 1, 1, 1 };
		CascadeTopology* top = new CascadeTopology(layerSizes, layerTypes);
		top->addLayerConnection(1, { 0 });
		top->addLayerConnection(2, { 1 });
		top->addLayerConnection(3, { 1, 2 });
		top->addLayerConnection(4, { 2, 3 });
		top->addLayerConnection(5, { 4 });
		top->addLayerConnection(6, { 5 });*/

		std::vector<int> layerSizes = { m_number_of_inputs + 16, 32, 16, m_number_of_outputs + 16};
		std::vector<int> layerTypes = { Layer::inputLayer, 1, 1, 1 };
		LayeredTopology* top = new LayeredTopology(layerSizes, layerTypes);


		m_controller = new LayeredNeuralNet(top); //memory is managed by network
		m_controller->initializeXavier();

		Individual::m_genome = std::shared_ptr<NeuralNetGenome>(new NeuralNetGenome(m_controller));
	}


	void destroyNeuralController() {
		if (m_controller)
			delete m_controller;
		m_controller = nullptr;
	}

public:
	NeuralNetChromosome() = delete;
	NeuralNetChromosome(const NeuralNetChromosome& copy_this) = delete;

	NeuralNetChromosome(NeuralNetChromosome&& move_this) = delete;
	NeuralNetChromosome& operator=(NeuralNetChromosome&& move_this) = delete;
};