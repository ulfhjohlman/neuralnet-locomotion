#pragma once
#include "RecurrentNeuralNet.h"
class ContinousTimeRecurrentNeuralNet : public RecurrentNeuralNet
{
public:

	ContinousTimeRecurrentNeuralNet() = default;

	virtual ~ContinousTimeRecurrentNeuralNet() = default;
};

typedef ContinousTimeRecurrentNeuralNet CRTNN;
