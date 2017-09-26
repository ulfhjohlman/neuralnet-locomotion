#pragma once
#include "Layer.h"
#include "math.h"

class RectifiedLinearUnitLayer : public Layer
{
public:
	RectifiedLinearUnitLayer() = default;
	RectifiedLinearUnitLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~RectifiedLinearUnitLayer() = default;

	virtual void input(const MatrixType& x) {
		Layer::input(x);

		//Maybe add different model here
		m_outputs = m_outputs.array().max(0);
		//m_outputs = std::move(m_outputs.array().max(0)); //compiler smart enough?
	}

	virtual void setRandom()
	{
		Layer::setRandom();
		m_weights.array() *= sqrt(2/m_weights.rows());  //Xavier initialization for relu
	}

	virtual void backprop(const MatrixType& backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		Layer::updateGradients(backpass_gradients.array() * m_outputs_pre_activation.array().max(0),
			prev_layer_outputs);
	}
private:
	/*  give these their own layer implementations
	//ScalarType alpha; //hyperparameter eg. leaky f(x) = max(ax, x);
	//std dist for noisy relu
	*/
};
