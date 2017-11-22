#pragma once
#include "InputLayer.h"
#include <string>

class ScalingLayer : public NeuralNet {
public:
	ScalingLayer() = default;
	ScalingLayer(int input_size, int cols) {
		m_scale_vector.resize(input_size, cols);
		m_scale_vector.setOnes();
	}
	virtual ~ScalingLayer() = default;
	ScalingLayer(const ScalingLayer& copy_this) {
		this->m_scale_vector = copy_this.m_scale_vector;
	}
	ScalingLayer(ScalingLayer&& move_this) {
		this->m_scale_vector.swap(move_this.m_scale_vector);
	}
	ScalingLayer& operator=(const ScalingLayer& copy_this) {
		this->m_scale_vector.resizeLike(copy_this.m_scale_vector);
		this->m_scale_vector = copy_this.m_scale_vector;
		return *this;
	}

	void setScaling(const MatrixType& scale_vector) {
		m_scale_vector = scale_vector;
	}

	MatrixType& getScaling() {
		return m_scale_vector;
	}

	ScalarType& operator()(const Eigen::Index index) {
		return m_scale_vector(index);
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

typedef ScalingLayer Receptors;