#pragma once

#include "FeedForwardNeuralNet.h"
#include "CascadeTopology.h"
#include <numeric>
#include <set>

class CascadeNeuralNet : public FeedForwardNeuralNet
{
public:
	CascadeNeuralNet() = default;
	CascadeNeuralNet(CascadeTopology * topology) : m_topology(topology) {
		FeedForwardNeuralNet::setTopology(topology);
		constructFromTopology();
	}
	virtual ~CascadeNeuralNet() = default;

	virtual void constructFromTopology() override {
		clearMembers();
		checkTopology(m_topology);

		Layer inputLayer(m_topology->getLayerSize(0), 0);
		m_layers.push_back(std::move(inputLayer));
		m_numberOfLayerInputs.push_back(0);
		
		size_t numberOfLayers = m_topology->getNumberOfLayers();
		//construct network from topology
		for (int i = 1; i < numberOfLayers; i++) {
			int numberOfInputs = getLayerInputs(i);
			m_numberOfLayerInputs.push_back(numberOfInputs);

			int layerSize = m_topology->getLayerSize(i);
			checkLayerArgs(layerSize, numberOfInputs);

			Layer newLayer(layerSize, numberOfInputs);
			m_layers.push_back(std::move(newLayer));
		}
	}

	virtual void input(const MatrixType& x) override {
		checkEmptyNetwork();
		const int numberOfCols = x.cols();

		//Load first layer
		m_layers.front().output() = x;//Remove this copy in future

		MatrixType xi;
		//Propagate forward
		for (int i = 1; i < m_layers.size(); i++) {
			xi.resize(m_numberOfLayerInputs[i], numberOfCols);
			loadInput(i, xi, numberOfCols);
			m_layers[i].input(xi);
		}
	}

protected:
	void checkTopology(CascadeTopology * top) {
#ifdef _NEURALNET_DEBUG
		if (top == nullptr) throw std::invalid_argument("topology nullptr");

		for (int i = 0; i < m_topology->getNumberOfLayers(); i++) {
			if (m_topology->getLayerConnections(i).empty()) {
				throw NeuralNetException("Layer with no connections to it.");
			}
			const std::vector<int>& connections = m_topology->getLayerConnections(i);
			//Copy to set, which is both unique and sorted.
			std::set<int> unique_elements(connections.begin(), connections.end()); 
			if (unique_elements.size() < connections.size()) {
				throw NeuralNetException("Multiple connections to same layer.");
			}
		}
#endif
	}

private:
	int getLayerInputs(int i)
	{
		int numberOfInputs = 0;
		for (const auto& layerIndex : m_topology->getLayerConnections(i))
			numberOfInputs += m_topology->getLayerSize(layerIndex);

		return numberOfInputs;
	}

	void loadInput(int i, MatrixType &xi, int numberOfCols) {
		int row = 0;
		for (const auto& layerIndex : m_topology->getLayerConnections(i)) {
			auto numberOfRows = m_layers[layerIndex].output().rows();
			//Unnecessary copy, if we can operate on weights matrix in chunks instead.
			xi.block(row, 0, numberOfRows, numberOfCols) = m_layers[layerIndex].output();
			row += numberOfRows;
		}
	}
	void clearMembers()
	{
		auto numberOfLayers = m_topology->getNumberOfLayers();

		m_numberOfLayerInputs.clear();
		m_numberOfLayerInputs.reserve(numberOfLayers);

		m_layers.clear();
		m_layers.reserve(numberOfLayers);
	}


	CascadeTopology* m_topology;
	std::vector<int> m_numberOfLayerInputs;
};
