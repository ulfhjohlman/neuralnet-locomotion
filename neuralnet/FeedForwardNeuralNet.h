#pragma once
#include "NeuralNet.h"
#include "Topology.h"
#include "LayeredTopology.h"
#include "Layer.h"

#include <vector>
#include <stdexcept>
#include <memory>


class FeedForwardNeuralNet :
	public NeuralNet
{
public:
	FeedForwardNeuralNet() : m_topology(nullptr) { }
	FeedForwardNeuralNet(LayeredTopology * topology) : m_topology(nullptr) {
		setTopology(topology);
		constructFromTopology();
	}
	virtual ~FeedForwardNeuralNet() { 
		if (m_topology)
			delete m_topology;
		destroyLayers();

	}

	virtual void setTopology(LayeredTopology* topology) {
		checkTopology(topology); 
		//new topology ok.
		//delete old if there is one.
		if (m_topology) delete m_topology;
		m_topology = topology;
	}

	virtual void constructFromTopology() {
		destroyLayers();
		m_layers.reserve(m_topology->getNumberOfLayers());

;
		Layer* inputLayer = new Layer(m_topology->getLayerSize( 0 ), 0);
		m_layers.push_back(inputLayer);

		//construct network from topology
		for (int i = 1; i < m_topology->getNumberOfLayers(); i++) {
			int numberOfInputs = m_topology->getLayerSize(i - 1);
			int layerSize = m_topology->getLayerSize(i);
			checkLayerArgs(layerSize, numberOfInputs);

			Layer* newLayer = new Layer(layerSize, numberOfInputs);
			m_layers.push_back(newLayer);
		}
	}

	virtual void initializeRandomWeights() {
		for (auto & layer : m_layers)
			layer->setRandom();
	}

	virtual void input(const MatrixType& x) {
		checkEmptyNetwork();

		//Load input layer 
		m_layers.front()->output() = x;//Remove this copy in future

		//Propagate forward
		for (int i = 1; i < m_layers.size(); i++) {
			m_layers[i]->input(m_layers[i-1]->output());
		}
	}

	virtual MatrixType& output() {
		checkEmptyNetwork();

		return m_layers.back()->output();
	}

	virtual void save(const char* toFile) { }
	virtual void load(const char* fromFile) { }
protected:
	void checkEmptyNetwork() const {
#ifdef _NEURALNET_DEBUG
		if (m_layers.empty()) throw NeuralNetException("Empty neural net");
#endif // _NEURALNET_DEBUG
	}
	void checkLayerArgs(int layerSize, int numberOfInputs)
	{
#ifdef _NEURALNET_DEBUG
		if (numberOfInputs < 1)
			throw NeuralNetException("Layer with no inputs.");
		if (layerSize < 1)
			throw NeuralNetException("Layer with no neurons.");
#endif // _NEURALNET_DEBUG
	}

	void destroyLayers()
	{
		for (size_t i = 0; i < m_layers.size(); i++)
			if (m_layers[i]) {
				//std::cout << "deleted:" << m_layers[i] << std::endl;
				delete m_layers[i];
			}
		m_layers.clear();
	}

	std::vector<Layer*> m_layers;//replace with shared_ptr<Layer>
private:
	void checkTopology(LayeredTopology* topology)
	{
#ifdef _NEURALNET_DEBUG
		if (topology == nullptr) throw std::invalid_argument("topology nullptr");
		if (topology->getNumberOfLayers() < 2) throw std::invalid_argument("input not specified.");
		if (topology == m_topology) return;
#endif // _NEURALNET_DEBUG
	}


	LayeredTopology* m_topology;
};

typedef FeedForwardNeuralNet FFNN;

