#pragma once
#include "Layer.h"

class SoftmaxLayer : public Layer
{
public:
	SoftmaxLayer() = default;
	SoftmaxLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~SoftmaxLayer() = default;

	virtual void input(const MatrixType& x) {
		Layer::input(x);
		m_outputs = m_outputs.array().exp();
		VectorType normFactor = m_outputs.colwise().sum(); //note: is column vector
		//Broadcasting divison incase of batch input
		m_outputs = m_outputs.array().rowwise() / normFactor.transpose().array();
	}
	virtual void backprop(const MatrixType& backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		Layer::updateGradients(backpass_gradients.array() * m_outputs.array(),
			prev_layer_outputs);
	}
};
