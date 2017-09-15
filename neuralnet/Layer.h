#pragma once
//This library
#include "NeuralNet.h"

//STL
#include <sstream>



// ViennaCL headers

#ifndef VIENNACL_WITH_OPENCL
#define VIENNACL_WITH_OPENCL
#endif

#ifndef VIENNACL_HAVE_EIGEN
#define VIENNACL_HAVE_EIGEN
#endif // !VIENNACL_HAVE_EIGEN

#include "viennacl/vector.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/ocl/backend.hpp"

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
		m_weights.array() = m_weights.array() * 10;
		m_weights.array() = m_weights.array().floor();
		//m_bias.setRandom();
		m_bias.setZero();
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
		
		//Add separate neuron activation here in the future
		//m_outputs.array() = m_outputs.array().tanh();

		std::cout << m_weights << std::endl;
		std::cout << x << std::endl << "=\n";
		std::cout << m_outputs << std::endl;

		std::cout << std::endl;
	}

	virtual MatrixType& output() {
		return m_outputs;
	}

	virtual void save(const char* toFile) { }
	virtual void load(const char* fromFile) { }
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

private:
	MatrixType m_weights;
	MatrixType m_outputs;
	VectorType m_bias;

	viennacl::matrix<ScalarType> m_gpu_weights; //Appends bias
	viennacl::vector<ScalarType> m_gpu_outputs;
};