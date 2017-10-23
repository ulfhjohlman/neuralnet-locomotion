#pragma once
#include "Layer.h"

class InputLayer : public Layer
{
public:
	InputLayer() = default;
	InputLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~InputLayer() = default;

	virtual void input(const MatrixType& x) {
		m_x = x;
	}

	virtual const MatrixType& output()
	{
		return m_x;
	}

	virtual void cachePushBackOutputs()
	{
		m_outputs_cache.push_back(m_x);
	}

	virtual void clearOutputsCache()
	{
		m_outputs_cache.clear();
	}

	virtual void uncacheIndexedOutput(int i)
	{
		m_x = m_outputs_cache[i];
	}
private:
	MatrixType m_x;
};
