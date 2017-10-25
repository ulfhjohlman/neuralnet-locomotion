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
            double valuefunc_mean_batch = 0;
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
                valuefunc_mean_batch += valueFuncMean();

                // add data to batch lists
                storeBatchData();

                //reset stuff
                env->reset();
            }
            mean_return_batch/=batch_size;
            std::cout << "Batch of " << batch_size <<
                " trajectories generated. Mean return over batch: " << mean_return_batch << "\n";
            std::cout << "ValueNet estimate:\t" << valuefunc_mean_batch << "\n";
            std::cout << "Optimizing over trajectories\n";
            for(int epoch=0; epoch < num_epochs; epoch++)
            std::cout << "Epoch: " << epoch << "\n";
            {
                for(int traj=0; traj < batch_size ; traj++)
                {
                    //cache loss gradients
                    calcAndBackpropGradients(traj);
                    // double meanLoss = calcLoss(traj);

                    double meanVFLoss = calcValueFuncLoss();
                    double meanVFpre = valueFuncMean();


                    //if end of a minibatch
                    if((traj+1) % mini_batch_size == 0)
                    {
					    policy.popCacheLayerParams();
				        policy.updateParams();
                        valueFunc.popCacheLayerParams();
                        valueFunc.updateParams();
                        //log progress
                        printf(" ---- Mini Batch Update %d ---- \n", (traj+1) / mini_batch_size);


                    }
                    // double meanLosspost = calcLoss(traj);
                    makeValuePredictions();
                    double meanVFLosspost = calcValueFuncLoss2();
                    double meanVFpost = valueFuncMean();
                    // std::cout << "PreLoss: \t" << meanLoss << "Post loss: \t" << meanLosspost << "\n";
                    std::cout << "PrevfLoss: \t" << meanVFLoss << "PostVF loss: \t" << meanVFLosspost << "\n";
                    std::cout << "PrevfMean: \t" << meanVFpre << "PostVF mean: \t" << meanVFpost << "\n";

                }


            }
            //clearLists();
            clearBatchLists();

        }
    }

protected:
    virtual void resizeLists(int len)
    {
        PolicyGradientTrainer::resizeLists(len);
        valuePred_list.resize(len);
        valueTarg_list.resize(len);
		valueFuncLoss_list.resize(len);
		valueFuncLoss_list2.resize(len);
		valueTargTD1.resize(len);
    }
    virtual void clearLists()
    {
        PolicyGradientTrainer::clearLists();
        valuePred_list.clear();
        valueTarg_list.clear();
		valueFuncLoss_list.clear();
		valueFuncLoss_list2.clear();
		valueTargTD1.clear();
    }
	void clearBatchLists()
	{
		batch_adv_list.clear();
		batch_ac_list.clear();
		batch_ob_list.clear();
		batch_prob_list.clear();
		batch_mu_list.clear();
		batch_vpred_list.clear();
		batch_vtarg_list.clear();
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
    double valueFuncMean()
    {
        double x=0;
        for (int i = 0; i<m_timesteps_per_episode; i++)
        {
            x+= valuePred_list[i];
        }
        return x;
    }

    //calc loss for trajectory i. returns mean loss
    double calcLoss(int i)
    {
        ScalarType meanLoss = 0;
        double ratio;
        for(int j = 0; j < m_timesteps_per_episode; j++)
        {
            ratio = calcPolicyRatio(i,j);
            loss_list[j] = - lClippObjFunc(ratio, batch_adv_list[i][j]);
            meanLoss+=loss_list[j];
        }
        meanLoss /= m_timesteps_per_episode;
        return meanLoss;
    }

	double calcValueFuncLoss()
	{
		ScalarType meanLoss = 0;
		for (int j = 0; j < m_timesteps_per_episode; j++)
		{
			// batch_valueFuncLoss_list[j] = (batch_valuePred_list[i][j] - batch_valueTarg_list[i][j]) *
			// (batch_valuePred_list[i][j] - batch_valueTarg_list[i][j]);
			valueFuncLoss_list[j] = (valuePred_list[j] - valueTarg_list[j]) *
				(valuePred_list[j] - valueTarg_list[j]);
			meanLoss += valueFuncLoss_list[j];
		}
		meanLoss /= m_timesteps_per_episode;
		return meanLoss;
	}
	double calcValueFuncLoss2()
	{
		ScalarType meanLoss = 0;
		for (int j = 0; j < m_timesteps_per_episode; j++)
		{
			// batch_valueFuncLoss_list[j] = (batch_valuePred_list[i][j] - batch_valueTarg_list[i][j]) *
			// (batch_valuePred_list[i][j] - batch_valueTarg_list[i][j]);
			valueFuncLoss_list2[j] = (valuePred_list[j] - valueTarg_list[j]) *
				(valuePred_list[j] - valueTarg_list[j]);
			meanLoss += valueFuncLoss_list2[j];
		}
		meanLoss /= m_timesteps_per_episode;
		return meanLoss;
	}

    //produce generalized advantage estimates using value network and state list
    void GAE(){
        // dont forget to first makeValuePredictions();
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
            
        }
		// could instead use TD(1):
		// valueTarg_list[i] = sum[from i to end] of rew_list;
		valueTargTD1[adv_list.size() - 1] = rew_list[adv_list.size()-1];
		for (int i = adv_list.size() - 2; i >= 0; i--)
		{
			valueTargTD1[i] = rew_list[i] + m_gamma*valueTargTD1[i + 1];
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
				//reforward pass for correct "input gradients"
				policy.forwardpassObs(batch_ob_list[i][j]);
                Eigen::Map<MatrixType> actionMatrix(batch_ac_list[i][j].data(),action_space_dim,1);
                Eigen::Map<MatrixType> muMatrix(batch_mu_list[i][j].data(),action_space_dim,1);
                // NEGATION because obejctive func = - loss func
                policy.backprop(- batch_adv_list[i][j] * r * ( actionMatrix.array() - muMatrix.array() )/policy.getSigma());
                policy.cacheLayerParams();

                Eigen::Matrix<ScalarType,1,1> vFuncGrad;
                vFuncGrad(0,0) = 2* (batch_vpred_list[i][j] - batch_vtarg_list[i][j]);
				//reforward pass for correct "input gradients"
				Eigen::Map<MatrixType> ob_matrix(batch_ob_list[i][j].data(), state_space_dim, 1);
				valueFunc.predict(ob_matrix);
				valueFunc.backprop( vFuncGrad );
                valueFunc.cacheLayerParams();
				std::cout << "Vpred: " << batch_vpred_list[i][j] << "\tVtarg: " << batch_vtarg_list[i][j] << "\tReal: " << valueTargTD1[j] << "\tOb:  " << batch_ob_list[i][j][0] << "," << batch_ob_list[i][j][1] << "\n";
            }
        }
    }

    //ratio pi(a|s)/pi_old(a|s) for batched trajectory i timestep j
    double calcPolicyRatio(int i, int j)
    {
        double new_policy_prob = policy.probAcGivenState(batch_ac_list[i][j], batch_ob_list[i][j]);
        double old_policy_prob = batch_prob_list[i][j];
        return new_policy_prob / old_policy_prob;
    }

    virtual void updateOldPolicy()
    {
        oldPolicy.copyParams(policy);
    }

    void storeBatchData()
    {
        batch_adv_list.push_back(std::move(adv_list));
        batch_ac_list.push_back(std::move(ac_list));
        // batch_ob_list.push_back(std::move(ob_list));
        batch_ob_list.push_back(ob_list);
        batch_prob_list.push_back(std::move(prob_list));
        batch_mu_list.push_back(std::move(mu_list));
        // batch_vpred_list.push_back(std::move(valuePred_list));
        // batch_vtarg_list.push_back(std::move(valueTarg_list));
        batch_vpred_list.push_back(valuePred_list);
        batch_vtarg_list.push_back(valueTarg_list);
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
	std::vector<ScalarType> valueFuncLoss_list;
	std::vector<ScalarType> valueFuncLoss_list2;
	std::vector<ScalarType> valueTargTD1;

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
