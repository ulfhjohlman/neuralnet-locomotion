#pragma once

#include "FeedForwardNeuralNet.h"
#include "CascadeTopology.h"
#include <numeric>

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

		Layer inputLayer(m_topology->getLayerSize(0), 0);
		m_layers.push_back(std::move(inputLayer));
		m_numberOfLayerInputs.push_back(0);
		
		int numberOfLayers = m_topology->getNumberOfLayers();
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

		//Load first layer
		m_layers.front().output() = x;//Remove this copy in future

		//Propagate forward
		for (int i = 1; i < m_layers.size(); i++) {
			MatrixType xi;
			int numberOfCols = m_layers[i - 1].output().cols();

			xi.resize(m_numberOfLayerInputs[i], numberOfCols);
			loadInput(i, xi, numberOfCols);

			m_layers[i].input(xi);
		}
	}

protected:


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
			int numberOfRows = m_layers[layerIndex].output().rows();
			xi.block(row, 0, numberOfRows, numberOfCols) = m_layers[layerIndex].output();
			row += numberOfRows;
		}
	}
	void clearMembers()
	{
		int numberOfLayers = m_topology->getNumberOfLayers();

		m_numberOfLayerInputs.clear();
		m_numberOfLayerInputs.reserve(numberOfLayers);

		m_layers.clear();
		m_layers.reserve(numberOfLayers);
	}


	CascadeTopology* m_topology;
	std::vector<int> m_numberOfLayerInputs;
};
