#pragma once
#include "InputLayer.h"
#include <string>

class ScalingLayer : public Layer {
public:
	ScalingLayer(int input_size, int cols) {
		m_scale_vector.resize(input_size, cols);
	}
	virtual ~ScalingLayer() = default;

	void setScaling(const MatrixType& scale_vector) {
		m_scale_vector = scale_vector;
	}
	void operator()(const Eigen::Index row, const Eigen::Index col, ScalarType x) {
		m_scale_vector(row, col) = x;
	}

	virtual void input(const MatrixType& x) {
		m_scaled_input = m_scale_vector.array() *  x.array();
	}
	virtual const MatrixType& output() {
		return m_scaled_input;
	}
private:
	MatrixType m_scale_vector;
	MatrixType m_scaled_input;
	//MatrixType m_variance;
	//MatrixType m_mean;
	//std::vector<std::string> m_scale_description;
};