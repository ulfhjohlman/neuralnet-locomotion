#pragma once
#include "RecurrentNeuralNet.h"
class ContinousTimeRecurrentNeuralNet : public RecurrentNeuralNet
{
public:

	ContinousTimeRecurrentNeuralNet()
	{
	}

	virtual ~ContinousTimeRecurrentNeuralNet()
	{
	}
};

typedef ContinousTimeRecurrentNeuralNet CRTNN;

