#pragma once
#ifdef  _DEBUG 
#define _NEURALNET_DEBUG //Use this for error checking in subclasses.
#endif //  _DEBUG

#include "Dataset.h"
#include "NeuralNetException.h"
#include "Eigen/dense"
typedef Eigen::MatrixXf MatrixType;
typedef Eigen::VectorXf VectorType;
typedef float			ScalarType;

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

	virtual void input(const MatrixType&) = 0;
	virtual MatrixType& output() = 0;

	/*virtual void setDataset(const Dataset&) = 0;
	virtual void train() = 0;*/

	virtual void save(const char* toFile) = 0;
	virtual void load(const char* fromFile) = 0;
};

