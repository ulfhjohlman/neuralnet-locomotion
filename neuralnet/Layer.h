prev_layer_outputs#pragma once
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

	Layer(const Layer& copy_this) :
		m_weights( copy_this.m_weights ),
		m_outputs( copy_this.m_outputs ),
	 	m_bias( copy_this.m_bias),
		m_outputs_pre_activation( copy_this.m_outputs_pre_activation),
		m_gradients_inputs( copy_this.m_gradients_inputs),
		m_gradients_weights( copy_this.m_gradients_weights),
		m_gradients_bias(copy_this.m_gradients_bias)  { }

	Layer(Layer&& move_this) {
		m_weights.swap(move_this.m_weights);
		m_outputs.swap(move_this.m_outputs);
		m_bias.swap(move_this.m_bias);
		m_outputs_pre_activation.swap(move_this.m_outputs_pre_activation);
		m_gradients_weights.swap(move_this.m_gradients_weights);
		m_gradients_inputs.swap(move_this.m_gradients_inputs);
		m_gradients_bias.swap(move_this.m_gradients_bias);
	}

	Layer& operator=(const Layer& copy_this) = delete;
	Layer& operator=(Layer&& move_this) = delete;

	virtual void setRandom() {
		m_weights.setRandom();
		m_bias.setRandom();
		m_outputs.setZero();
		m_gradients_weights.setZero();
		m_gradients_inputs.setZero();
		m_gradients_bias.setZero();
		m_outputs_pre_activation.setZero();
	}

	virtual void setLayer(int size, int inputSize) {
		checkSize(size);
		checkSize(inputSize);

		m_outputs.resize(size, 1);
		m_outputs_pre_activation.resizeLike(m_outputs);

		m_weights.resize(size, inputSize);
		m_bias.resize(size);

		m_gradients_weights.resizeLike(m_weights);
		m_gradients_bias.resizeLike(m_bias);
		m_gradients_inputs.resize(inputSize,1);

	}

	virtual void input(const MatrixType& x) {
		checkNeuronMismatch(m_outputs, m_weights, x);

		//Load input onto layer and compute
		//No aliasing issues here.
		m_outputs.noalias() = m_weights * x;

		//Add bias to each neuron
		m_outputs.array().colwise() += m_bias.array();

		m_outputs_pre_activation = m_outputs;

		//Subclass for separate neuron activation here

		//printOperations(x);
	}
	virtual void backprop(const MatrixType& backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		// backpass_gradients is a vector of dL/dY where L is scalar loss
		// and Y a vector of this layers outputs
		// prev_layer_outputs == this layers inputs
		updateGradients(backpass_gradients.array() * m_outputs_pre_activation.array(), prev_layer_outputs);
	}

	void updateWeights(float learning_rate)
	{
		// updates weights proportionaly to their currently stored gradients
		m_weights += learning_rate * m_gradients_weights;
		m_bias += learning_rate * m_gradients_bias;
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
		softmax = 6,
		noLayer = 7 //always last for errorChecking
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
	inline void checkNeuronMismatch(const MatrixType& y, const MatrixType& A, const MatrixType& x) {
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

	void updateGradients(const MatrixType& modified_backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		//modified_backpass_gradients is activationlayer dependant
		checkNeuronMismatch(m_gradients_inputs,m_weights,modified_backpass_gradients);
		checkNeuronMismatch(m_gradients_weights,modified_backpass_gradients,prev_layer_outputs.transpose());
		checkMatchingSize(m_gradients_bias,modified_backpass_gradients);
		m_gradients_inputs.noalias() = m_weights * modified_backpass_gradients;
		m_gradients_weights.noalias() = modified_backpass_gradients * prev_layer_outputs.transpose();
		m_gradients_bias.noalias() = modified_backpass_gradients;
	}

	MatrixType m_outputs;
	MatrixType m_outputs_pre_activation;
private:
	MatrixType m_weights;
	VectorType m_bias;

	MatrixType m_gradients_weights;
	MatrixType m_gradients_inputs;
	VectorType m_gradients_bias;
};
