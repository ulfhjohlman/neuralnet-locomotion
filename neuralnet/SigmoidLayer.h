#pragma once
#include "Layer.h"

class SigmoidLayer : public Layer
{
public:
	SigmoidLayer() = default;
	SigmoidLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~SigmoidLayer() = default;

	virtual void input(const MatrixType& x) {
		Layer::input(x);
		m_outputs = logisticFunc(m_outputs);
	}
	static MatrixType logisticFunc(const MatrixType& x)
	{
		return (static_cast<ScalarType>(1) / (static_cast<ScalarType>(1) + (-x.array()).exp()));
	}
	virtual void backprop(const MatrixType& backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		Layer::updateGradients(backpass_gradients.array() *(m_outputs.array())*( static_cast<ScalarType>(1)-m_outputs.array()),
			prev_layer_outputs);
	}
};
