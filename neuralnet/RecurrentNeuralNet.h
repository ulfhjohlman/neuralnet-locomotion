#pragma once
#include "LayeredNeuralNet.h"
#include "RecurrentTopology.h"

class RecurrentNeuralNet :
	public LayeredNeuralNet
{
public:
	RecurrentNeuralNet() : LayeredNeuralNet(), m_topology(nullptr) { }
	RecurrentNeuralNet(RecurrentTopology* topology) : m_topology(topology) { }
	virtual ~RecurrentNeuralNet() = default;

private:
	RecurrentTopology* m_topology;
};

typedef RecurrentNeuralNet RNN;
