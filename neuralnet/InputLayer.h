#pragma once
#include "Layer.h"

class InputLayer : public Layer
{
public:
	InputLayer() = default;
	InputLayer(int size, int inputSize) : Layer(size, inputSize) { }
	virtual ~InputLayer() = default;

	virtual void input(const MatrixType& x) {
		//m_pInput = &x;
		m_x = x;
	}

	virtual const MatrixType& output() {
		//return *m_pInput;
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
	virtual void save(const char* path);
	virtual void load(const char* path);

	virtual ScalarType* weightData() {
		throw NeuralNetException("Attempted to get weight from a input layer.");
	}
	virtual ScalarType* biasData() {
		throw NeuralNetException("Attempted to get bias from a input layer.");
	}

protected:
	MatrixType m_x;
	//MatrixType const* m_pInput;
};

void InputLayer::save(const char* path) {
	std::string data_structure_name = "layer_" + std::to_string(0);
	m_document.insert(data_structure_name.c_str());

	//size and type
	m_document.insertAttribute("layer_type", static_cast<int>(Layer::inputLayer));
	m_document.insertAttribute("rows_output_size", static_cast<int>(m_outputs.rows()));
	m_document.insertAttribute("cols_input_size", static_cast<int>(m_weights.cols()));

	NeuralNet::save((path + data_structure_name).c_str());
}

void InputLayer::load(const char* path) {
	NeuralNet::load(path);
	int rows, cols;
	m_document.getAttribute("rows_output_size", rows);
	m_document.getAttribute("cols_input_size", cols);
	if (cols != static_cast<int>(0) || rows != static_cast<int>(m_outputs.rows()))
		throw NeuralNetException("Load error, matrix mismatch");
}