#pragma once
#include "NeuralNet.h"
#include "stdlib.h"

class PolicyWrapper
{
public:
        PolicyWrapper(NeuralNet * new_nn, int new_in_size, int new_out_size): nn(new_nn), in_size(new_in_size) {}

        //returns a single action_space output from the policy given an observation
        std::vector<ScalarType>& samplePolicy(std::vector<ScalarType> obs)
        {
            Eigen::Map<MatrixType> in_matrix(obs.data(),in_size,1);
            nn->input(in_matrix);
            out_matrix_ptr = &nn->output();
            sample = std::vector<ScalarType>(out_matrix_ptr->data(), out_matrix_ptr->data() + out_matrix_ptr->size());
            return sample;
        }
private:
        NeuralNet * nn;
        std::vector<ScalarType> sample;
        MatrixType in_matrix;
        const MatrixType * out_matrix_ptr = nullptr;
        int in_size;
        int out_size;

};
