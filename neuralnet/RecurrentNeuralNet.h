#pragma once
#include "NeuralNet.h"

class RecurrentNeuralNet :
	public NeuralNet
{
public:
	RecurrentNeuralNet() = default;
	virtual ~RecurrentNeuralNet() = default;
};

typedef RecurrentNeuralNet RNN;
