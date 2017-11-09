#pragma once
#ifdef  _DEBUG
#define _NEURALNET_DEBUG //Use this for error checking in subclasses.
#endif //  _DEBUG

#include <string>
#include "XMLFile.h"
#include "NeuralNetException.h"
#include "../lib/Eigen/Dense"
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
	virtual const MatrixType& output() = 0;

	virtual void save(const char* toFile);
	virtual void load(const char* fromFile);
	virtual void setName(const char* name);
	//virtual void setDescription(const char* descripion);

protected:
	XMLFile m_document;
	std::string m_name;
};

void NeuralNet::save(const char* toFile) {
	m_document.save(toFile);
	m_document.print();
}

void NeuralNet::load(const char* fromFile) {
	m_document.load(fromFile);
	m_document.print();
}

void NeuralNet::setName(const char* name) {
	m_name = name;
}