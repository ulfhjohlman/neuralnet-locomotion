#pragma once
#include "Layer.h"

class HyperbolicTangentLayer : public Layer
{
public:
	HyperbolicTangentLayer() = default;
	HyperbolicTangentLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~HyperbolicTangentLayer() = default;

	virtual void input(const MatrixType& x) {
		Layer::input(x);

		//Maybe add beta here
		m_outputs.array() = m_outputs.array().tanh();
	}

	virtual void backprop(const MatrixType& backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		// d/dx(tanh(x)) = 1-tanh^2(x)
		updateGradients(backpass_gradients.array() * (1-m_outputs.array().square()),
		 			prev_layer_outputs);
	}
private:
	ScalarType m_beta; //tanh(beta*x)
};
