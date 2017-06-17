#pragma once
#ifdef  _DEBUG 
#define _NEURALNET_DEBUG //Use this for error checking in subclasses.
#endif //  _DEBUG




#include "Dataset.h"

class NeuralInput;
class NeuralOutput;

/// <summary>
/// Basic interface for neural net.
/// </summary>
class NeuralNet
{
public:
	NeuralNet() = default;
	virtual ~NeuralNet() = default;

	virtual void feed(const NeuralInput&) = 0;
	virtual void getOutput(NeuralOutput&) = 0;

	virtual void setDataset(const Dataset&) = 0;
	virtual void train() = 0;

	virtual void save(const char* toFile) = 0;
	virtual void load(const char* fromFile) = 0;
};

