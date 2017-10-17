#pragma once
#include "NeuralNet.h"
#include "LayeredNeuralNet.h"
#include "stdlib.h"
#include "Generator.h"
#define _USE_MATH_DEFINES //for M_PI
#include <math.h>

// Wrapps a neuralnet into the context of a reinforcement learning policy function
class PolicyWrapper
{
public:
        PolicyWrapper(LayeredNeuralNet * new_nn, int new_in_size, int new_out_size): nn(new_nn), in_size(new_in_size) {
            sample.reserve(new_out_size);
        }

        //returns a single action_space output from the policy given an observation
        const std::vector<ScalarType>& samplePolicy(std::vector<ScalarType> obs)
        {
            Eigen::Map<MatrixType> in_matrix(obs.data(),in_size,1);
            nn->input(in_matrix);
            out_matrix_ptr = &nn->output();
            mu = std::vector<ScalarType>(out_matrix_ptr->data(), out_matrix_ptr->data() + out_matrix_ptr->size());
            sample.clear();
            ScalarType x;
            total_prob = 1;
            for(int i=0;i<mu.size();i++)
            {
                x = generator.generate_normal<ScalarType>(mu[i],m_sigma*m_sigma);
                sample.push_back(x);
                //store probability of the sample in 'prob'
                total_prob *= norm_pdf(x,mu[i],m_sigma);
            }
            calcLocalErrorGradient();
            return sample;
        }

        virtual void input(const MatrixType& x)
        {
            nn->input(x);
        }

        //returns the cumulative probability of the sample
        const double getCumulativeProb()
        {
            return total_prob;
        }

        const std::vector<ScalarType>& getLocalErrorGradient()
        {
            return localErrorGradient;
        }

        void backprop(MatrixType err_gradients)
        {
            nn->backprop(err_gradients);
        }

        void cacheLayerParams()
        {
            nn->cacheLayerParams();
        }

		void popCacheLayerParams()
		{
			nn->popCacheLayerParams();
		}
		void updateParams()
		{
			nn->updateParameters();
		}

private:
        void calcLocalErrorGradient()
        {
            localErrorGradient.clear();
            for(int i =0;i<mu.size();i++)
            {
                localErrorGradient.push_back((mu[i] - sample[i])/(m_sigma*m_sigma));
            }
        }

        ScalarType norm_pdf( ScalarType x , ScalarType mu ,ScalarType sigma)
        {
            return 1.0/(sigma*sqrt(2*M_PI)) * exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma));
        }

        LayeredNeuralNet * nn;
        std::vector<ScalarType> sample;
        std::vector<ScalarType> localErrorGradient;
        double total_prob;
        std::vector<ScalarType> mu;
        MatrixType in_matrix;
        const MatrixType * out_matrix_ptr = nullptr;
        int in_size;
        int out_size;
        Generator generator; //Thread safe generation
        double m_sigma = 1;
};
