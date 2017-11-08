#pragma once
#include "NeuralNet.h"
#include "LayeredNeuralNet.h"
#include "stdlib.h"
#include "Generator.h"
#include "SigmoidLayer.h"
#define _USE_MATH_DEFINES //for M_PI
#include <math.h>

/*** 
 Wrapps a neuralnet into the context of a reinforcement learning policy function. 
 The NN outputs mu and sigma to parametrize a gaussian distribution
 For n output distributions 2*n parameters are needed. The network outputs are ordered as : [mu_1 mu_2 mu_3 sigma_1 sigma_2 sigma_3]
 The final layer type of a PolicyWrapper net should be without activation function. It handles these on a per parameter basis by itself (te.x. sigmoid for std)
 ***/

class PolicyWrapper
{
public:
        PolicyWrapper(LayeredNeuralNet * new_nn, int new_in_size, int new_out_size): m_nn(new_nn), in_size(new_in_size), out_size(new_out_size){
            sample.reserve(new_out_size);
			localSigmaObjectiveGradient.reserve(new_out_size);
			localMuObjectiveGradient.reserve(new_out_size);
			sigma.reserve(new_out_size);
			mu.reserve(new_out_size);
#ifdef _DEBUG
			int lastLayerIndex = new_nn->getTopology()->getNumberOfLayers() - 1;
			if (new_nn->getTopology()->getLayerSize(lastLayerIndex) != 2*new_out_size)
			{
				throw std::invalid_argument("Policy Network needs 2*action_dim outputs!  ( ==2*out_size)\n");
			}
			if(new_nn->getTopology()->getLayerType(lastLayerIndex) != Layer::noActivation)
			{
				throw std::invalid_argument("Policy Network needs no activation functions on final layer. It handles them itself!\n");
			}
#endif
			// the out_size is the dimensionality of the samples produced
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
                x = generator.generate_normal<ScalarType>(mu[i],sigma[i]);
                sample.push_back(x);
                //store probability of the sample in 'prob'
                total_prob *= norm_pdf(x,mu[i],sigma[i]);

            }
            calcLocalObjectiveGradients();
            return sample;
        }
        void forwardpassObs(std::vector<ScalarType>& obs)
        {
            Eigen::Map<MatrixType> in_matrix(obs.data(),in_size,1);
            m_nn->input(in_matrix);
            out_matrix_ptr = &m_nn->output();
			mu = std::vector<ScalarType>(out_matrix_ptr->data(), out_matrix_ptr->data() + out_matrix_ptr->size()/2);
			sigma = std::vector<ScalarType>(out_matrix_ptr->data() + out_matrix_ptr->size()/2, out_matrix_ptr->data() + out_matrix_ptr->size());
			for (int i = 0; i < sigma.size(); i++)
			{
				sigma[i] = 1 / (1 + exp(-sigma[i]));
				if (sigma[i] < 1e-8)
				{
					sigma[i] = 1e-8; //dont allow sigma = 0
					#ifdef _DEBUG
						std::cout << "Warning: sigma close to zero in PolicyWrapper!\n";
					#endif	
				}
			}
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
                prob *= norm_pdf(action[i],mu[i],sigma[i]);
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
		const std::vector<ScalarType>& getSigma()
		{
			return sigma;
		}
		const std::vector<ScalarType>& getlocalMuObjectiveGradient()
		{
			return localMuObjectiveGradient;
		}
		const std::vector<ScalarType>& getlocalSigmaObjectiveGradient()
		{
			return localSigmaObjectiveGradient;
		}

        void backprop(MatrixType err_gradients)
        {
			for (int i = 0; i < sigma.size(); i++)
			{
				err_gradients(i + out_size / 2, 0) *= sigma[i]*(1-sigma[i]);
			}
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

        void copyParams(const PolicyWrapper& otherPW)
        {
            m_nn->copyParams(*otherPW.m_nn);
        }


private:
        void calcLocalObjectiveGradients()
        {
			localMuObjectiveGradient.clear();
			localSigmaObjectiveGradient.clear();
            for(int i =0;i<out_size;i++)
            {
				localMuObjectiveGradient.push_back((sample[i] - mu[i]) / (sigma[i]*sigma[i]));
				localSigmaObjectiveGradient.push_back((sample[i] - mu[i])*(sample[i] - mu[i]) / (sigma[i]* sigma[i]* sigma[i]) - 1/(sigma[i]));
            }
        }

        ScalarType norm_pdf( ScalarType x , ScalarType local_mu ,ScalarType local_sigma)
        {
            return 1.0/(local_sigma*sqrt(2*M_PI)) * exp(-(x-local_mu)*(x-local_mu)/(2.0*local_sigma*local_sigma));
        }

protected:
        std::vector<ScalarType> sample;
		std::vector<ScalarType> localMuObjectiveGradient;
		std::vector<ScalarType> localSigmaObjectiveGradient;
        double total_prob;
		std::vector<ScalarType> mu;
		std::vector<ScalarType> sigma;

        MatrixType in_matrix;
        const MatrixType * out_matrix_ptr = nullptr;
        int in_size;
        int out_size;
        Generator generator; //Thread safe generation
       

        LayeredNeuralNet * m_nn;
};
