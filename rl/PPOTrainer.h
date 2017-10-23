#pragma once
#include "PolicyGradientTrainer.h"
#include "ValueFuncWrapper.h"
#include <math.h>

class PPOTrainer : public PolicyGradientTrainer
{
public:
    PPOTrainer(Environment * new_env, LayeredNeuralNet * new_policy, LayeredNeuralNet * new_oldPolicy, LayeredNeuralNet * new_valueFunc)
            : PolicyGradientTrainer(new_env, new_policy), oldPolicy(new_oldPolicy,state_space_dim,action_space_dim), valueFunc(new_valueFunc,state_space_dim)  {}

    void set_GAE_lambda(double new_lambda)
    {
        m_lambda = new_lambda;
    }

    void trainPPO(int max_iterations, int batch_size , int timesteps_per_episode, int mini_batch_size,int num_epochs)
    {
        m_timesteps_per_episode = timesteps_per_episode;
        for(int iteration = 0; iteration < max_iterations; iteration++)
        {
            double mean_return_batch = 0;
            updateOldPolicy();
            reserveBatchLists(batch_size);
            for(int traj=0; traj < batch_size ; traj++)
            {
                //generate a trajectory (an episode)
                resizeLists(timesteps_per_episode); //redo each loop as the lists are std::moved at storeDataBatch()
                generateTrajectory(timesteps_per_episode);
                makeValuePredictions();
                GAE();
                standardizeVector(adv_list);

                mean_return_batch += episode_return;

                // add data to batch lists
                storeBatchData();

                //reset stuff
                env->reset();
            }
            mean_return_batch/=batch_size;
            std::cout << "Batch of " << batch_size <<
                " trajectories generated. Mean return over batch: " << mean_return_batch << "\n";
            std::cout << "Optimizing over trajectories\n";
            for(int epoch=0; epoch < num_epochs; epoch++)
            {
                for(int traj=0; traj < batch_size ; traj++)
                {
                    //cache loss gradients
                    calcAndBackpropGradients(traj);


                    //if end of a minibatch
                    if((traj+1) % mini_batch_size == 0)
                    {
					    policy.popCacheLayerParams();
				        policy.updateParams();
                        valueFunc.popCacheLayerParams();
                        valueFunc.updateParams();
                        //log progress
                        printf(" ---- Mini Batch Update %d ---- \n", traj+1 / mini_batch_size);


                    }
                }


            }
            //clearLists();
            //clearBatchLists();

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

    void reserveBatchLists(int size)
    {
        batch_adv_list.reserve(size);
        batch_ac_list.reserve(size);
        batch_ob_list.reserve(size);
        batch_prob_list.reserve(size);
        batch_mu_list.reserve(size);
        batch_vpred_list.reserve(size);
        batch_vtarg_list.reserve(size);
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
    // Calculate the loss gradients over the entire trajectory i
    virtual void calcAndBackpropGradients(int i)
    {
        for(int j=0; j < m_timesteps_per_episode; j++)
        {
            double r = calcPolicyRatio(i,j);
            if((r * batch_adv_list[i][j]) > (batch_adv_list[i][j]*clipRelativeOne(r, eps)) && ((r > 1+eps) || (r<1-eps)))
            {
                // then derivative = 0
                continue;
            }
            else //backprop grads
            {
                Eigen::Map<MatrixType> actionMatrix(batch_ac_list[i][j].data(),action_space_dim,1);
                Eigen::Map<MatrixType> muMatrix(batch_mu_list[i][j].data(),action_space_dim,1);
                // NEGATION because obejctive func = - loss func
                policy.backprop(- batch_adv_list[i][j] * r * ( actionMatrix.array() - muMatrix.array() )/policy.getSigma());
                policy.cacheLayerParams();

                Eigen::Matrix<ScalarType,1,1> vFuncGrad;
                vFuncGrad(0,0) = 2* (batch_vpred_list[i][j] - batch_vtarg_list[i][j]);
                valueFunc.backprop( vFuncGrad );
                valueFunc.cacheLayerParams();
            }
        }
    }

    //ratio pi(a|s)/pi_old(a|s) for batched timestep i
    double calcPolicyRatio(int i, int j)
    {
        double new_policy_prob = policy.probAcGivenState(batch_ac_list[i][j], batch_ob_list[i][j]);
        double old_policy_prob = batch_prob_list[i][j];
        return new_policy_prob / old_policy_prob;
    }

    virtual void updateOldPolicy()
    {
        throw std::runtime_error("NOT IMPLEMENTED\n");
    }

    void storeBatchData()
    {
        batch_adv_list.push_back(std::move(adv_list));
        batch_ac_list.push_back(std::move(ac_list));
        batch_ob_list.push_back(std::move(ob_list));
        batch_prob_list.push_back(std::move(prob_list));
        batch_mu_list.push_back(std::move(mu_list));
        batch_vpred_list.push_back(std::move(valuePred_list));
        batch_vtarg_list.push_back(std::move(valueTarg_list));
    }

    ScalarType lClippObjFunc(ScalarType ratio, ScalarType adv)
    {
        return fmin( ratio * adv, clipRelativeOne(ratio, eps) * adv);
        //can try std::min in <alorithms> but it should be slower
    }

    //clips r to be within [1-eps,1+eps]
    ScalarType clipRelativeOne(ScalarType r, ScalarType epsilon)
    {
        return clip(r,1.0-epsilon,1.0+epsilon);
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
    PolicyWrapper oldPolicy;
    std::vector<ScalarType> valuePred_list;
    std::vector<ScalarType> valueTarg_list;

    int m_timesteps_per_episode=0;
    double m_lambda = 0.95;
    double eps = 0.2;

    // lists of data over the full batch
    std::vector<std::vector<ScalarType>>                batch_adv_list;
    std::vector<std::vector<std::vector<ScalarType>>>   batch_ac_list;
    std::vector<std::vector<std::vector<ScalarType>>>   batch_ob_list;
    std::vector<std::vector<double>>                    batch_prob_list;
    std::vector<std::vector<std::vector<ScalarType>>>   batch_mu_list;
    std::vector<std::vector<ScalarType>>                batch_vpred_list;
    std::vector<std::vector<ScalarType>>                batch_vtarg_list;

};
