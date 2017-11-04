#pragma once
#include "NeuralNet.h"
#include "LayeredNeuralNet.h"
#include "stdlib.h"
#include "Generator.h"
#define _USE_MATH_DEFINES //for M_PI
#include <math.h>

// Wrapps a neuralnet into the context of a reinforcement learning policy function. 
// The NN outputs mu to parametrize a gaussian distribution with a given sigma
class PolicyWrapper
{
public:
        PolicyWrapper(LayeredNeuralNet * new_nn, int new_in_size, int new_out_size): m_nn(new_nn), in_size(new_in_size) {
            sample.reserve(new_out_size);
        }
        

        //returns a single action_space output from the policy given an observation
        const std::vector<ScalarType>& samplePolicy(std::vector<ScalarType>& obs)
        {
            forwardpassObs(obs);
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
            calcLocalMuObjectiveGradient();
            return sample;
        }
        void forwardpassObs(std::vector<ScalarType>& obs)
        {
            Eigen::Map<MatrixType> in_matrix(obs.data(),in_size,1);
            m_nn->input(in_matrix);
            out_matrix_ptr = &m_nn->output();
            mu = std::vector<ScalarType>(out_matrix_ptr->data(), out_matrix_ptr->data() + out_matrix_ptr->size());
        }

        virtual void input(const MatrixType& x)
        {
            m_nn->input(x);
        }
        double probAcGivenState(std::vector<ScalarType> action , std::vector<ScalarType> obs)
        {
            forwardpassObs(obs);
            double prob = 1;
            for(int i=0;i<mu.size();i++)
            {
                prob *= norm_pdf(action[i],mu[i],m_sigma);
            }
            return prob;
        }
        //returns the cumulative probability of the sample
        const double getCumulativeProb()
        {
            return total_prob;
        }
        const std::vector<ScalarType>& getMu()
        {
            return mu;
        }
        const std::vector<ScalarType>& getlocalMuObjectiveGradient()
        {
            return localObjectiveGradient;
        }

        void backprop(MatrixType err_gradients)
        {
            m_nn->backprop(err_gradients);
        }

        void cacheLayerParams()
        {
            m_nn->cacheLayerParams();
        }

		void popCacheLayerParams()
		{
			m_nn->popCacheLayerParams();
		}
		void updateParams()
		{
			m_nn->updateParameters();
		}

        double getSigma()
        {
            return m_sigma;
        }
		void setSigma(double new_sigma)
		{
			if(!(new_sigma > 0)) {
				throw std::invalid_argument("Sigma in gausian distribution must be positive!\n");
			}
			m_sigma = new_sigma;
		}
        void copyParams(const PolicyWrapper& otherPW)
        {
            m_nn->copyParams(*otherPW.m_nn);
        }


private:
        void calcLocalMuObjectiveGradient()
        {
            localObjectiveGradient.clear();
            for(int i =0;i<mu.size();i++)
            {
                localObjectiveGradient.push_back((sample[i]-mu[i])/(m_sigma*m_sigma));
            }
        }

        ScalarType norm_pdf( ScalarType x , ScalarType mu ,ScalarType sigma)
        {
            return 1.0/(sigma*sqrt(2*M_PI)) * exp(-(x-mu)*(x-mu)/(2.0*sigma*sigma));
        }

protected:
        std::vector<ScalarType> sample;
        std::vector<ScalarType> localObjectiveGradient;
        double total_prob;
        std::vector<ScalarType> mu;
        MatrixType in_matrix;
        const MatrixType * out_matrix_ptr = nullptr;
        int in_size;
        int out_size;
        Generator generator; //Thread safe generation
        double m_sigma = 1;

        LayeredNeuralNet * m_nn;
};
