#pragma once
//This library
#include "NeuralNet.h"

//STL
#include <sstream>

class Layer : public NeuralNet
{
public:
	Layer() = default;
	Layer(int size, int inputSize) {
		setLayer(size, inputSize);
	}
	virtual ~Layer() = default;

	Layer(const Layer& copy_this) : m_weights( copy_this.m_weights ), m_outputs( copy_this.m_outputs ){
		//std::cout << "Layer copy constructor\n";
	}

	Layer(Layer&& move_this) {
		//std::cout << "Layer move constructor\n";

		m_weights.swap(move_this.m_weights);
		m_outputs.swap(move_this.m_outputs);
		m_bias.swap(move_this.m_bias);
	}

	Layer& operator=(const Layer& copy_this) = delete;
	Layer& operator=(Layer&& move_this) = delete;

	virtual void setRandom() {
		m_weights.setRandom();
		m_bias.setRandom();
		m_outputs.setZero();
	}

	virtual void setLayer(int size, int inputSize) {
		checkSize(size);
		checkSize(inputSize);

		m_outputs.resize(size, 1);
		m_weights.resize(size, inputSize);
		m_bias.resize(size);
	}

	virtual void input(const MatrixType& x) {
		checkNeuronMismatch(m_outputs, m_weights, x);

		//Load input onto layer and compute
		//No aliasing issues here.
		m_outputs.noalias() = m_weights * x;

		//Add bias to each neuron
		m_outputs.array().colwise() += m_bias.array();

		//Subclass for separate neuron activation here
		m_outputs.array() = m_outputs.array().tanh() / 2;

		//printOperations(x);
	}

	virtual MatrixType& output() {
		return m_outputs;
	}

	virtual void save(const char* toFile) { }
	virtual void load(const char* fromFile) { }

	enum LayerType
	{
		noActivation = 0,
		tanh = 1,
		sigmoid = 2,
		relu = 3,
		inputLayer = 4,
		scalingLayer = 5,
		noLayer = 6 //always last for errorChecking
	};
protected:

	inline void checkSize(int size) {
#ifdef _NEURALNET_DEBUG
		if (size < 0) { throw NeuralNetException("Must be >=0 sized layer\n"); }
#endif // _NEURALNET_DEBUG
	}

	inline void checkMatchingSize(const MatrixType& lhs, const MatrixType& rhs) {
#ifdef _NEURALNET_DEBUG
		bool colsNotEqual = lhs.cols() != rhs.cols();
		bool rowsNotEqual = lhs.rows() != rhs.rows();
		if (colsNotEqual || rowsNotEqual) {
			std::ostringstream os;
			os << " Not matching matrix size (rows,cols) " << __LINE__ << " at " << __FILE__ << std::endl;
			throw NeuralNetException( os.str().c_str() );
		}
#endif // _NEURALNET_DEBUG
	}

	//y=A*x
	inline void checkNeuronMismatch(const MatrixType& y, MatrixType& A, const MatrixType& x) {
#ifdef _NEURALNET_DEBUG
		bool neuronMismatch = A.cols() != x.rows(); //y = A*x, y in R^(wxm), A in R^(wxn) , x in R^(nxm)
		bool resultMismatch = y.cols() != x.cols();
		if (neuronMismatch) {
			std::ostringstream os;
			os << " Weights do not match input " << A.cols() << " != " << x.rows() << std::endl;
			throw NeuralNetException(os.str().c_str());
		}
		else if (resultMismatch) {
			std::ostringstream os;
			os << " y = A*x, y not right dimension, gpu will fail. "<< std::endl;
			std::cout << os.str();
			//throw NeuralNetException(os.str().c_str());
		}
#endif
	}

	void printOperations(const MatrixType& x)
	{
		std::cout << "format: A\n x \n +\n b\n =\n y\n";
		std::cout << m_weights << std::endl;
		std::cout << x << "+\n" << m_bias << std::endl << "=\n";
		std::cout << m_outputs << std::endl;
		std::cout << std::endl;
	}
private:
	MatrixType m_weights;
	MatrixType m_outputs;
	VectorType m_bias;
};
