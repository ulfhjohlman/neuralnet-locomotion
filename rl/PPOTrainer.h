#pragma once
#include "PolicyGradientTrainer.h"
#include "ValueFuncWrapper.h"
#include <math.h>

class PPOTrainer : public PolicyGradientTrainer
{
public:
    PPOTrainer(Environment * new_env, LayeredNeuralNet * new_policy, LayeredNeuralNet * new_valueFunc)
            : PolicyGradientTrainer(new_env, new_policy), valueFunc(new_valueFunc,state_space_dim)  {}

    void set_GAE_lambda(double new_lambda)
    {
        m_lambda = new_lambda;
    }

    virtual void train(int max_episodes, int timesteps_per_episode, int batch_size)
    {
        resizeLists(timesteps_per_episode);
        double mean_loss;
		double mean_loss_batch = 0;
		double mean_return_batch = 0;
        for(int i=1; i <= max_episodes; i++)
        {
            //generate a trajectory (an episode)
            generateTrajectory(timesteps_per_episode,true);

            //at end of an episode
            {
                makeValuePredictions();
                GAE();
				// mean_loss = calcLoss();
                // mean_loss_batch += mean_loss;
				mean_return_batch += episode_return;
				// printf("Episode: %d. \tMean loss: %lf\n", i, mean_loss);

                calcGradients();

                backpropGradients();
                //cache loss gradients

                //if end of a minibatch
                if(i % batch_size == 0 && i>0)
                {
                    mean_return_batch/=batch_size;
					mean_loss_batch = mean_loss_batch/ static_cast<double>(batch_size);
					policy.popCacheLayerParams();
				    policy.updateParams();
                    valueFunc.popCacheLayerParams();
                    valueFunc.updateParams();
                    //log progress
					if((i/batch_size) % 10 == 0){
                        printf(" ---- Batch Update %d ---- \tmean return: %lf \tmean loss over batch: %lf \n", i / batch_size, mean_return_batch, mean_loss_batch);
                    }
					mean_loss_batch = 0;
					mean_return_batch = 0;

                }


                //reset stuff
                env->reset();
                //clearLists();
            }

        }
    }

protected:
    virtual void resizeLists(int len)
    {
        PolicyGradientTrainer::resizeLists(len);
        valuePred_list.resize(len);
        valueTarg_list.resize(len);
    }
    virtual void clearLists()
    {
        PolicyGradientTrainer::clearLists();
        valuePred_list.clear();
        valueTarg_list.clear();
    }

    //produces the value prediction lists using the valueFunction network
    void makeValuePredictions()
    {
        for(int i = 0 ; i<ob_list.size() ; i++)
        {
            Eigen::Map<MatrixType> ob_matrix(ob_list[i].data(),state_space_dim,1);
            valuePred_list[i] = valueFunc.predict(ob_matrix)(0,0);
        }
    }

    //produce generalized advantage estimates using value network and state list
    void GAE(){
        makeValuePredictions();
        double delta = rew_list.back() - valuePred_list.back();
        adv_list.back() = delta;

        for(int i=adv_list.size()-2;i>=0;i--)
        {
            delta = rew_list[i] + m_gamma * valuePred_list[i+1] - valuePred_list[i];
            adv_list[i] = delta + m_lambda*m_gamma*adv_list[i+1];
        }
        for(int i=0;i<adv_list.size();i++)
        {
            valueTarg_list[i] = valuePred_list[i] + adv_list[i]; //TD_lambda residual error
            // could instead use TD(1):
            // valueTarg_list[i] = sum[from i to end] of rew_list;
        }
    }

    virtual void calcGradients()
    {
        throw std::runtime_error("NOT IMPLEMENTED\n");
    }

    virtual void backpropGradients()
    {
        throw std::runtime_error("NOT IMPLEMENTED\n");
    }


    ScalarType lClippObjFunc(ScalarType pi_old, ScalarType pi_new, ScalarType adv)
    {
        ScalarType ratio = pi_new/pi_old;
        return fmin( ratio * adv, clipRelativeOne(ratio, eps) * adv);
        //can try std::min in <alorithms> but it should be slower
    }

    //clips r to be within [1-eps,1+eps]
    ScalarType clipRelativeOne(ScalarType r, ScalarType eps)
    {
        return clip(r,1.0-eps,1.0+eps);
    }

    //clips x if outside interval
    ScalarType clip(ScalarType x, ScalarType lower_bound, ScalarType upper_bound)
    {
        if(x > upper_bound){
            return upper_bound;
        }
        if(x < lower_bound){
            return lower_bound;
        }
        return x;
    }

    ValueFuncWrapper valueFunc;
    std::vector<ScalarType> valuePred_list;
    std::vector<ScalarType> valueTarg_list;
    double m_lambda = 0.95;
    double eps = 0.2;
};
