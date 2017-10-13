#pragma once
#include "NeuralNet.h"
#include "stdlib.h"
#include "Generator.h"
#include <math.h>

class PolicyWrapper
{
public:
        PolicyWrapper(NeuralNet * new_nn, int new_in_size, int new_out_size): nn(new_nn), in_size(new_in_size) {
            sample.reserve(out_size);
            prob.reserve(out_size);
        }

        //returns a single action_space output from the policy given an observation
        std::vector<ScalarType>& samplePolicy(std::vector<ScalarType> obs)
        {
            Eigen::Map<MatrixType> in_matrix(obs.data(),in_size,1);
            nn->input(in_matrix);
            out_matrix_ptr = &nn->output();
            mu = std::vector<ScalarType>(out_matrix_ptr->data(), out_matrix_ptr->data() + out_matrix_ptr->size());
            sample.clear();
            prob.clear();
            ScalarType x;
            for(int i=0;i<mu.size();i++)
            {
                //with sigma = 1 for all variables; (for now)
                x = generator.generate_normal<ScalarType>(mu[i],1);
                sample.push_back(x);
                //store probability of the sample in 'prob'
                prob.push_back(norm_pdf(x,mu[i],1));
            }
            return sample;
        }

        //returns the cumulative probability of the sample TODO: BETTER
        double getCumulativeProb()
        {

        }

private:
        ScalarType norm_pdf( ScalarType x , mu , sigma)
        {
            return 1.0/(sigma*sqrt(2*M_PI)) * exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma))
        }

        NeuralNet * nn;
        std::vector<ScalarType> sample;
        std::vector<ScalarType> prob;
        std::vector<ScalarType> mu;
        MatrixType in_matrix;
        const MatrixType * out_matrix_ptr = nullptr;
        int in_size;
        int out_size;
        Generator generator; //Thread safe generation
};
