#pragma once
#include "FeedForwardNeuralNet.h"
#include "RecurrentTopology.h"

class RecurrentNeuralNet :
	public FeedForwardNeuralNet
{
public:
	RecurrentNeuralNet() : FeedForwardNeuralNet(), m_topology(nullptr) {  };
	RecurrentNeuralNet(RecurrentTopology* topology) : m_topology(topology) { }
	virtual ~RecurrentNeuralNet() = default;

private:
	RecurrentTopology* m_topology;
};

typedef RecurrentNeuralNet RNN;
