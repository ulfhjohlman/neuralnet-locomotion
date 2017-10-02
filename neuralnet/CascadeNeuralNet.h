#pragma once

#include "LayeredNeuralNet.h"
#include "CascadeTopology.h"
#include <numeric>
#include <set>

/// <summary>
/// Blev ej super logiskt. FeedForwardNerualNet �r egentligen
/// ett subset av denna klass konceptuellt. Ska fixa senare
/// </summary>
class CascadeNeuralNet : public LayeredNeuralNet
{
public:
	CascadeNeuralNet() = default;
	CascadeNeuralNet(CascadeTopology * topology) : m_topology(topology) {
		LayeredNeuralNet::setTopology(topology);
		constructFromTopology();
	}
	virtual ~CascadeNeuralNet() = default;

	virtual void constructFromTopology() override {
		clearMembers();
		checkTopology(m_topology);

		Layer* inputLayer = LayerFactory::constructLayer(m_topology->getLayerSize(0), 0, m_topology->getLayerType(0));
		m_layers.push_back(inputLayer);
		m_numberOfLayerInputs.push_back(0);

		size_t numberOfLayers = m_topology->getNumberOfLayers();
		//construct network from topology
		for (int i = 1; i < numberOfLayers; i++) {
			int numberOfInputs = getLayerInputs(i);
			int layerSize = m_topology->getLayerSize(i);
			int layerType = m_topology->getLayerType(i);

			Layer* newLayer = LayerFactory::constructLayer(layerSize, numberOfInputs, layerType);
			m_layers.push_back(newLayer);
			m_numberOfLayerInputs.push_back(numberOfInputs);
		}
	}

	virtual void input(const MatrixType& x) override {
		checkEmptyNetwork();
		const int numberOfCols = x.cols();

		//Load first layer
		m_layers.front()->input(x);

		MatrixType xi; //This should be fixed in future
		//Propagate forward
		for (int i = 1; i < m_layers.size(); i++) {
			xi.resize(m_numberOfLayerInputs[i], numberOfCols);
			loadInput(i, xi, numberOfCols);
			m_layers[i]->input(xi);
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
			auto numberOfRows = m_layers[layerIndex]->output().rows();
			//Unnecessary copy, if we can operate on weights matrix in chunks instead.
			xi.block(row, 0, numberOfRows, numberOfCols) = m_layers[layerIndex]->output();
			row += numberOfRows;
		}
	}

	void clearMembers()
	{
		size_t numberOfLayers = m_topology->getNumberOfLayers();

		m_numberOfLayerInputs.clear();
		m_numberOfLayerInputs.reserve(numberOfLayers);

		destroyLayers();
		m_layers.reserve(numberOfLayers);
	}

	CascadeTopology* m_topology;
	std::vector<int> m_numberOfLayerInputs;
};
