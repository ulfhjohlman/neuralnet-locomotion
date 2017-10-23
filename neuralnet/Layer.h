#pragma once
//This library
#include "NeuralNet.h"

//Utility
#include "Generator.h"

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
		//m_outputs_pre_activation( copy_this.m_outputs_pre_activation),
		m_gradients_inputs( copy_this.m_gradients_inputs),
		m_gradients_weights( copy_this.m_gradients_weights),
		m_gradients_bias(copy_this.m_gradients_bias)  { }

	Layer(Layer&& move_this) {
		m_weights.swap(move_this.m_weights);
		m_outputs.swap(move_this.m_outputs);
		m_bias.swap(move_this.m_bias);
		//m_outputs_pre_activation.swap(move_this.m_outputs_pre_activation);
		m_gradients_weights.swap(move_this.m_gradients_weights);
		m_gradients_inputs.swap(move_this.m_gradients_inputs);
		m_gradients_bias.swap(move_this.m_gradients_bias);
	}

	Layer& operator=(const Layer& copy_this) = delete;
	Layer& operator=(Layer&& move_this) = delete;

	virtual void setRandom() {
		Generator generator; //Thread safe generation
		generator.fill_vector_normal<ScalarType>(m_weights.data(), m_weights.size(), 0, 1);
		generator.fill_vector_uniform<ScalarType>(m_bias.data(), m_bias.size(), -0.05, 0.05);

		m_outputs.setZero();
		m_gradients_weights.setZero();
		m_gradients_inputs.setZero();
		m_gradients_bias.setZero();

		m_gradients_weights_cache.setZero();
		m_gradients_bias_cache.setZero();
		//m_outputs_pre_activation.setZero();
	}
	virtual void setRandomXavier() {
		setRandom();
		m_weights.array() *= static_cast<ScalarType>(sqrt( 1.0 / m_weights.rows()));
	}

	virtual void setLayer(int size, int inputSize) {
		checkSize(size);
		checkSize(inputSize);

		m_outputs.resize(size, 1);
		//m_outputs_pre_activation.resizeLike(m_outputs);

		m_weights.resize(size, inputSize);
		m_bias.resize(size,1);

		m_gradients_weights.resizeLike(m_weights);
		m_gradients_bias.resizeLike(m_bias);
		m_gradients_inputs.resize(inputSize,1);

		m_gradients_weights_cache.resizeLike(m_weights);
		m_gradients_bias_cache.resizeLike(m_bias);
	}

	virtual void input(const MatrixType& x) {
		checkNeuronMismatch(m_outputs, m_weights, x);

		//Load input onto layer and compute
		//No aliasing issues here.
		m_outputs.noalias() = m_weights * x;

		//Add bias to each neuron
		m_outputs.array() += m_bias.array();

		//m_outputs_pre_activation = m_outputs;

		//Subclass for separate neuron activation here

		//printOperations(x);
	}
	virtual void backprop(const MatrixType& backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		// backpass_gradients is a vector of dL/dY where L is scalar loss
		// and Y a vector of this layers outputs
		// prev_layer_outputs == this layers inputs
		updateGradients(backpass_gradients, prev_layer_outputs);
	}

	//use vanila SGD to update weights
	void updateWeightsSGD(double learning_rate)
	{
		// updates weights proportionally to their currently stored gradients
		m_weights -= learning_rate * m_gradients_weights;
		m_bias -= learning_rate * m_gradients_bias;
	}

	friend class ParameterUpdater;

	virtual const MatrixType& output() {
		return m_outputs;
	}
	virtual const MatrixType& getInputGradients() {
		return m_gradients_inputs;
	}

	virtual ScalarType* weightData() {
		return m_weights.data();
	}
	virtual ScalarType* biasData() {
		return m_bias.data();
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
	const MatrixType& getWeights()
	{
		return m_weights;
	}
	const MatrixType& getBias()
	{
		return m_bias;
	}
	const MatrixType& getWeightGradients()
	{
		return m_gradients_weights;
	}

	virtual void cacheParamGradients()
	{
		m_gradients_weights_cache +=m_gradients_weights;
		m_gradients_bias_cache +=m_gradients_bias;
	}
	virtual void popCacheParamGradients()
	{
		m_gradients_weights = m_gradients_weights_cache;
		m_gradients_bias = m_gradients_bias_cache;
		m_gradients_weights_cache.setZero();
		m_gradients_bias_cache.setZero();
	}

	virtual void reserveOutputCache(int i)
	{
		m_outputs_cache.reserve(i);
	}

	virtual void cachePushBackOutputs()
	{
		m_outputs_cache.push_back(m_outputs); //stores a copy of m_outputs
	}

	virtual void clearOutputsCache()
	{
		m_outputs_cache.clear();
	}

	virtual void uncacheIndexedOutput(int i)
	{
		m_outputs = m_outputs_cache[i];
	}

protected:
	void updateGradients(const MatrixType& modified_backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		//modified_backpass_gradients is activation layer dependent
		checkNeuronMismatch(m_gradients_inputs.transpose(), modified_backpass_gradients.transpose(), m_weights);
		checkNeuronMismatch(m_gradients_weights, modified_backpass_gradients, prev_layer_outputs.transpose().eval());
		checkMatchingSize(m_gradients_bias, modified_backpass_gradients);

		// alternativly transpose weight matrix: (A*B)^T = A^T*B^T
		m_gradients_inputs.noalias() = (modified_backpass_gradients.transpose() * m_weights).transpose();
		m_gradients_weights.noalias() = modified_backpass_gradients * prev_layer_outputs.transpose();
		m_gradients_bias = modified_backpass_gradients;
	}
	void printOperations(const MatrixType& x)
	{
		std::cout << "format: A\n x \n +\n b\n =\n y\n";
		std::cout << m_weights << std::endl;
		std::cout << x << "+\n" << m_bias << std::endl << "=\n";
		std::cout << m_outputs << std::endl;
		std::cout << std::endl;
	}

protected: //members
	MatrixType m_outputs;
	//MatrixType m_outputs_pre_activation;
	MatrixType m_weights;
	std::vector<MatrixType> m_outputs_cache;
	
private: //members
	MatrixType m_bias;
	MatrixType m_gradients_weights;
	MatrixType m_gradients_inputs;
	MatrixType m_gradients_bias;
	MatrixType m_gradients_weights_cache;
	MatrixType m_gradients_bias_cache;


protected: //Error checking
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
		bool resultMismatch = y.cols() != x.cols() || y.rows() != A.rows();
		if (neuronMismatch) {
			std::ostringstream os;
			os << " Weights do not match input. A size: ("<<A.rows()<<","<<A.cols()<<"), X size: (" <<
					x.rows()<<","<<x.cols()<<"). " << A.cols() << " != " << x.rows() << std::endl;
			throw NeuralNetException(os.str().c_str());
		}
		else if (resultMismatch) {
			std::ostringstream os;
			os << " y = A*x, y not right dimension, gpu will fail. A size: (" << A.rows() << "," <<
					A.cols() << "), X size: (" << x.rows() << "," << x.cols() << "), Y size: (" <<
					y.rows() << "," << y.cols() << ")." << std::endl;
			std::cout << os.str();
			throw NeuralNetException(os.str().c_str());
		}
#endif
	}
};
