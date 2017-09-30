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
		m_outputs.array() = m_outputs.array().max(0);
		//m_outputs = std::move(m_outputs.array().max(0)); //compiler smart enough?
	}

	virtual void setRandom()
	{
		Layer::setRandom();
		m_weights.array() *= static_cast<ScalarType>
			(sqrt( 2.0 / m_weights.rows()));  //Xavier initialization for relu
	}

	virtual void backprop(const MatrixType& backpass_gradients, const MatrixType& prev_layer_outputs)
	{
		//MatrixType mask.array() = (m_outputs.array() > 0.0f);
		//auto masked_gradient = mask * backpass_gradients.array();
		//Layer::updateGradients(masked_gradient, prev_layer_outputs);
	    Layer::updateGradients((m_outputs.array() != 0).select(backpass_gradients,m_outputs), prev_layer_outputs); // is there a better function for this masking in eigen?
	}
private:
	/*  give these their own layer implementations
	//ScalarType alpha; //hyperparameter eg. leaky f(x) = max(ax, x);
	//std dist for noisy relu
	*/
};
