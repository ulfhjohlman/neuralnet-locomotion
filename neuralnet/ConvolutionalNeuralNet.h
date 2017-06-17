#pragma once
#include "NeuralNet.h"

class ConvolutionalNeuralNet : public NeuralNet
{
public:
	ConvolutionalNeuralNet() = default;
	virtual ~ConvolutionalNeuralNet() = default;
};

typedef ConvolutionalNeuralNet CNN