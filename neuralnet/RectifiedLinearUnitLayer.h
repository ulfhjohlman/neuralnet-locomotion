#pragma once
#include "Layer.h"

class RectifiedLinearUnitLayer : public Layer
{
public:
	RectifiedLinearUnitLayer() = default;
	RectifiedLinearUnitLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~RectifiedLinearUnitLayer() = default;

	virtual void input(const MatrixType& x) {
		Layer::input(x);

		//Maybe add different model here
		m_outputs.array().max(0);
	}
private:
	ScalarType alpha; //hyperparameter eg. leaky f(x) = max(ax, x);
	//std dist for noisy relu
};