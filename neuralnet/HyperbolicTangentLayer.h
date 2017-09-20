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
private:
	ScalarType m_beta; //tanh(beta*x)
};