#pragma once
#include "Layer.h"

class InputLayer : public Layer
{
public:
	InputLayer() = default;
	InputLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~InputLayer() = default;

	virtual void input(const MatrixType& x) {
		m_pInput = &x;
	}

	virtual const MatrixType& output()
	{
		return *m_pInput;
	}

private:
	MatrixType const* m_pInput;
};
