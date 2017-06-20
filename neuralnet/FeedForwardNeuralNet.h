#pragma once
#include "NeuralNet.h"
class FeedForwardNeuralNet :
	public NeuralNet
{
public:
	FeedForwardNeuralNet() = default;
	virtual ~FeedForwardNeuralNet() = default;
};

typedef FeedForwardNeuralNet FFNN;

