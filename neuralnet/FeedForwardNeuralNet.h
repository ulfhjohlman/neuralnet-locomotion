#pragma once
#include "NeuralNet.h"
#include "LayeredTopology.h"
#include "Layer.h"
#include <vector>



class FeedForwardNeuralNet :
	public NeuralNet
{
public:
	FeedForwardNeuralNet() : m_topology(nullptr) { m_layers.reserve(10); }
	FeedForwardNeuralNet(LayeredTopology * topology) : m_topology(topology) {
		if (topology == nullptr) return;
		if (topology->getNumberOfLayers() < 2) return;
		m_layers.reserve(topology->getNumberOfLayers());
		
		//construct network from topology
		for (int i = 1; i < topology->getNumberOfLayers(); i++) {
			int numberOfInputs = topology->getLayerSize(i - 1);
			int layerSize = topology->getLayerSize(i);
			
			m_layers.push_back(std::move(Layer(layerSize, numberOfInputs)));

			std::cout << "layer" << i << ": " << numberOfInputs*layerSize << " neurons." << std::endl;
		}
	}
	virtual ~FeedForwardNeuralNet() { 
		if (m_topology)
			delete m_topology;
	}

	virtual void initializeRandomWeights() {
		for (auto & layer : m_layers)
			layer.setRandom();
	}

	virtual void input(const MatrixType& x) {
		checkEmptyNetwork();

		//Load and compute first layer
		m_layers.front().input(x);

		//Propagate forward
		for (int i = 1; i < m_layers.size(); i++) {
			m_layers[i].input(m_layers[i-1].output());
		}
	}

	virtual MatrixType& output() {
		checkEmptyNetwork();

		return m_layers.back().output();
	}


	virtual void save(const char* toFile) {}
	virtual void load(const char* fromFile) {}
protected:
	void checkEmptyNetwork() const {
#ifdef _NEURALNET_DEBUG
		if (m_layers.empty()) throw NeuralNetException("Empty neural net");
#endif // _NEURALNET_DEBUG
	}
private:
	std::vector<Layer> m_layers;
	LayeredTopology * m_topology;
};

typedef FeedForwardNeuralNet FFNN;

