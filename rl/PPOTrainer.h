#pragma once
#include "PolicyGradientTrainer.h"
#include "ValueFuncWrapper.h"
#include <math.h>
#include <experimental/filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>

class PPOTrainer : public PolicyGradientTrainer
{
public:
    PPOTrainer(Environment * new_env, LayeredNeuralNet * new_policy, LayeredNeuralNet * new_valueFunc)
            : PolicyGradientTrainer(new_env, new_policy), valueFunc(new_valueFunc,state_space_dim)  {}

    void set_GAE_lambda(double new_lambda)
    {
        m_lambda = new_lambda;
    }

	//saves the current NN. will overwrite previous save.
	void save_NN() {
		std::experimental::filesystem::create_directory("best_net");
		policy.m_nn->save("best_net/");
		std::experimental::filesystem::create_directory("best_net/valuenet");
		valueFunc.m_nn->save("best_net/valuenet/");
	}

	void setName(const char* name) {
		policy.m_nn->setName(name);
	}

    void trainPPO(int max_iterations, int traj_batch_size , int timesteps_per_episode, int mini_batch_size,int num_epochs)
    {
		avg_returns_list.reserve(max_iterations);
		shuffle_order.reserve(traj_batch_size * timesteps_per_episode);
		value_batch_size = mini_batch_size;
		update_batch_size = mini_batch_size;
		m_timesteps_per_episode = timesteps_per_episode;
        for(int iteration = 0; iteration < max_iterations; iteration++)
        {
			policy.setLogSigma(-0.7 - 0.9*(iteration / static_cast<double>(max_iterations)));
			std::cout << "---- Iteration: " << iteration << " ----\n";
            double mean_return_batch = 0;
            //updateOldPolicy();
            reserveBatchLists(traj_batch_size);
            for(int traj=0; traj < traj_batch_size ; traj++)
            {
                //generate a trajectory (an episode)
                resizeLists(timesteps_per_episode); //redo each loop as the lists are std::moved at storeDataBatch()
                generateTrajectory(timesteps_per_episode);
				//standardizeVector(rew_list); 
				makeValuePredictions();
                //GAE();
				//simpleAdvEstimates();
				simpleAdvEstimates2();

				//TrainValueFunc();

                mean_return_batch += episode_return;
				//std::cout << "ValueFuncPred[0]:\t" << valuePred_list[0] << " TD1[0]:\t " << valueTargTD1[0] << " TD1[400]:\t " << valueTargTD1[400] << " Return: " << episode_return << "\n";
                // add data to batch lists
                storeBatchData(); //moves traj lists to their batch_list counterpart. note: they are std::moved and no longer accessable through old non-batchlists

                //reset stuff
                env->reset();
            }
            mean_return_batch /= traj_batch_size;
            std::cout << "Batch of " << traj_batch_size <<
                " trajectories generated. Mean return over batch: " << mean_return_batch << "\n";
			avg_returns_list.push_back(mean_return_batch);
			if (mean_return_batch > best_return) {
				std::cout << "New best_net saving...";
				save_NN();
				std::cout << " Save complete.\n";
				best_return = mean_return_batch;
				best_return_iteration = iteration;
			}

			std::cout << "Optimizing over trajectories\n";
			BatchTrainValueFunc(timesteps_per_episode);
			standardizeBatchVector(batch_adv_list);
			shuffle_order.clear();
			for (int epoch = 0; epoch < num_epochs; epoch++){
				for (int k = 0; k < traj_batch_size * timesteps_per_episode; k++) {
					shuffle_order.push_back(k);
				}
				std::random_shuffle(shuffle_order.begin(), shuffle_order.end());
				for (int k = 0; k < shuffle_order.size(); k++) {
					int i = shuffle_order[k] / timesteps_per_episode;
					int j = shuffle_order[k] % timesteps_per_episode;
					if (batch_traj_length[i] > j) {
						calcAndBackpropGradients(i, j, update_batch_size);
					}
				}
				//incase of residual updates
				policy.popCacheLayerParams();
				policy.updateParams();
			}
			std::cout << "Ratio of Zero/non-zero derivatives: " << static_cast<double>(zero_derivatives) / static_cast<double>(non_zero_derivatives) << "\n";
			zero_derivatives = 0;
			non_zero_derivatives = 0;
			double avgsigma = 0;
			for (int i = 0; i < batch_sigma_list[0][0].size(); i++) {
				avgsigma += (batch_sigma_list[0][0][i] / batch_sigma_list[0][0].size());
			}
			std::cout << "average non_log_sigma: " << avgsigma << " \n";
            //clearLists();
            clearBatchLists();

        }
		std::cout << "Optimization done. Best iteration was iter " << best_return_iteration << " with return " << best_return <<"!\n The corresponding net has been saved in ./best_net\n";
		save_tsv(avg_returns_list,"best_net/avg_mean_returns.tsv");
    }

protected:
    virtual void resizeLists(int len)
    {
        PolicyGradientTrainer::resizeLists(len);
        valuePred_list.resize(len);
        valueTarg_list.resize(len);
		valueTargTD1.resize(len);
    }
    virtual void clearLists()
    {
        PolicyGradientTrainer::clearLists();
        valuePred_list.clear();
        valueTarg_list.clear();
		valueTargTD1.clear();
    }
	void clearBatchLists()
	{
		batch_adv_list.clear();
		batch_ac_list.clear();
		batch_ob_list.clear();
		batch_prob_list.clear();
		batch_mu_list.clear();
		batch_sigma_list.clear();
		batch_vpred_list.clear();
		batch_vtarg_list.clear();
		batch_valueTargTD1.clear();
		batch_traj_length.clear();
	}
    void reserveBatchLists(int size)
    {
        batch_adv_list.reserve(size);
        batch_ac_list.reserve(size);
        batch_ob_list.reserve(size);
        batch_prob_list.reserve(size);
		batch_mu_list.reserve(size);
		batch_sigma_list.reserve(size);
        batch_vpred_list.reserve(size);
        batch_vtarg_list.reserve(size);
		batch_valueTargTD1.reserve(size);
		batch_traj_length.reserve(size);
    }

    //produces the value prediction lists using the valueFunction network
    void makeValuePredictions()
    {
        for(int i = 0 ; i<traj_length ; i++)
        {
            Eigen::Map<MatrixType> ob_matrix(ob_list[i].data(),state_space_dim,1);
            valuePred_list[i] = valueFunc.predict(ob_matrix)(0,0);
        }
    }

    //calc loss for trajectory i. returns mean loss
    double calcLoss(int i)
    {
        ScalarType meanLoss = 0;
        double ratio;
        for(int j = 0; j < batch_traj_length[i]; j++)
        {
            ratio = calcPolicyRatio(i,j);
            loss_list[j] = - lClippObjFunc(ratio, batch_adv_list[i][j]);
            meanLoss+=loss_list[j];
        }
        meanLoss /= batch_traj_length[i];
        return meanLoss;
    }

	void simpleAdvEstimates() {
		valueTargTD1[traj_length-1] = rew_list[traj_length-1];
		for (int i = traj_length - 2; i >= 0; i--)
		{
			valueTargTD1[i] = rew_list[i] + m_gamma*valueTargTD1[i + 1];
		}
		// same thing!
		for (int i = traj_length-1; i >= 0; i--)
		{
			adv_list[i] = valueTargTD1[i];
		}
	}
	void simpleAdvEstimates2() {
		valueTargTD1[traj_length - 1] = rew_list[traj_length - 1];
		adv_list[traj_length - 1] = valueTargTD1[traj_length - 1] - valuePred_list[traj_length - 1];
		if (isNaN(valuePred_list[traj_length - 1])) std::cout << "valuePred is NaN!\n";
		if (isNaN(valueTargTD1[traj_length - 1])) std::cout << "valueTargTD1 is NaN!\n";
		valueTarg_list[traj_length - 1] = valueTargTD1[traj_length - 1];
		for (int i = traj_length - 2; i >= 0; i--)
		{

			valueTargTD1[i] = rew_list[i] + m_gamma*valueTargTD1[i + 1];
			adv_list[i] = valueTargTD1[i] - valuePred_list[i];
			valueTarg_list[i] = valueTargTD1[i];
			if (isNaN(valuePred_list[i])) std::cout << "valuePred is NaN!\n";
			if (isNaN(valueTargTD1[i])) std::cout << "valueTargTD1 is NaN!\n";

		}
	}

	bool isNaN(double x) {
		return x != x;
	}

    //produce generalized advantage estimates using value network and state list
    void GAE(){
        // dont forget to first makeValuePredictions();
        double delta = rew_list[traj_length-1] - valuePred_list[traj_length-1];
        adv_list[traj_length-1] = delta;

        for(int i=traj_length-2;i>=0;i--)
        {
            delta = rew_list[i] + m_gamma * valuePred_list[i+1] - valuePred_list[i];
            adv_list[i] = delta + m_lambda*m_gamma*adv_list[i+1];
        }
		for (int i = 0; i < traj_length; i++)
		{
			valueTarg_list[i] = valuePred_list[i] + adv_list[i]; //TD_lambda residual error

		}
		// could instead use TD(1):
		// valueTarg_list[i] = sum[from i to end] of rew_list;
		valueTargTD1[traj_length - 1] = rew_list[traj_length - 1];
		for (int i = traj_length - 2; i >= 0; i--)
		{
			valueTargTD1[i] = rew_list[i] + m_gamma*valueTargTD1[i + 1];
		}
	}

	void TrainValueFunc()
	{
		for (int j = 0; j < traj_length; j++) {
			Eigen::Matrix<ScalarType, 1, 1> vFuncGrad;

			//reforward pass for correct "input gradients"
			Eigen::Map<MatrixType> ob_matrix(ob_list[j].data(), state_space_dim, 1);
			double valuePred = valueFunc.predict(ob_matrix)(0, 0);
			vFuncGrad(0, 0) = 2 * (valuePred - valueTargTD1[j]);
			valueFunc.backprop(vFuncGrad);
			valueFunc.cacheLayerParams();
			if ((j+1) % value_batch_size == 0) {
				valueFunc.popCacheLayerParams();
				valueFunc.updateParams();
			}
		}
		valueFunc.popCacheLayerParams();
		valueFunc.updateParams();
	}
	
	void BatchTrainValueFunc(int timesteps_per_episode)
	{
		int count = 0;
		for (int k = 0; k < shuffle_order.size(); k++) {
			int i = shuffle_order[k] / timesteps_per_episode; //traj
			int j = shuffle_order[k] % timesteps_per_episode; //timestep
			if (batch_traj_length[i] > j) {
				//reforward pass for correct "input gradients"
				Eigen::Map<MatrixType> ob_matrix(batch_ob_list[i][j].data(), state_space_dim, 1);
				double vPred  = valueFunc.predict(ob_matrix)(0, 0);

				Eigen::Matrix<ScalarType, 1, 1> vFuncGrad;
				vFuncGrad(0, 0) = 2 * (vPred - batch_vtarg_list[i][j]);

				valueFunc.backprop(vFuncGrad);
				valueFunc.cacheLayerParams();
				count++;
				if (count % value_batch_size == 0) {
					valueFunc.popCacheLayerParams();
					valueFunc.updateParams();
				}
			}
		}
		valueFunc.popCacheLayerParams();
		valueFunc.updateParams();
	}
	// Calculate the loss gradients over trajectory i step j
	virtual void calcAndBackpropGradients(int i, int j, int mini_batch_size)
	{
			double r = calcPolicyRatio(i, j);
			if (r > 100 || r < 0.01) "Oddly large r detected!\n";
			if ((r * batch_adv_list[i][j]) > (batch_adv_list[i][j] * clipRelativeOne(r, eps)) && ((r > 1 + eps) || (r < 1 - eps)))
			{
				// then derivative = 0
				zero_derivatives++;
			}
			else //backprop grads
			{
				//reforward pass for correct "input gradients"
				policy.forwardpassObs(batch_ob_list[i][j]);
				Eigen::Map<MatrixType> actionMatrix(batch_ac_list[i][j].data(), action_space_dim, 1);
				Eigen::Map<MatrixType> muMatrix(batch_mu_list[i][j].data(), action_space_dim, 1);
				Eigen::Map<MatrixType> sigmaMatrix(batch_sigma_list[i][j].data(), action_space_dim, 1);


				// NEGATION because obejctive func = - loss func
				MatrixType mu_error_gradients = -batch_adv_list[i][j] * r * (actionMatrix.array() - muMatrix.array()) / (sigmaMatrix.array().square());
				if (isNaN(batch_adv_list[i][j])) std::cout << "batch_adv_list[i][j] is NaN!\n";
				for (int qwe = 0; qwe < batch_ac_list[i][j].size(); qwe++) {
					if (isNaN(batch_ac_list[i][j][qwe])) std::cout << "batch_ac_list[i][j] is NaN!\n";
					if (isNaN(batch_mu_list[i][j][qwe])) std::cout << "batch_mu_list[i][j] is NaN!\n";
					if (isNaN(batch_sigma_list[i][j][qwe])) std::cout << "batch_sigma_list[i][j] is NaN!\n";
				}

				policy.backprop(mu_error_gradients);
				policy.cacheLayerParams();
				non_zero_derivatives++;
			}
			if (((++batch_backprop_count) % mini_batch_size) == 0)
			{
				batch_backprop_count = 0;
				policy.popCacheLayerParams();
				policy.updateParams();
			}
			//std::cout << "Pres update Obj: " << lClippObjFunc(r, batch_adv_list[i][j]) << "\n";
			//std::cout << "Post update Obj: " << lClippObjFunc(calcPolicyRatio(i, j), batch_adv_list[i][j]) << "\n";
	}

	//ratio pi(a|s)/pi_old(a|s) for batched trajectory i timestep j
	double calcPolicyRatio(int i, int j)
	{
		double new_policy_prob = policy.probAcGivenState(batch_ac_list[i][j], batch_ob_list[i][j]);
		double old_policy_prob = batch_prob_list[i][j];
		return new_policy_prob / old_policy_prob;
	}

	/*virtual void updateOldPolicy()
	{
		oldPolicy.copyParams(policy);
	}*/

	void storeBatchData()
	{
		batch_adv_list.push_back((adv_list));
		batch_ac_list.push_back((ac_list));
		batch_ob_list.push_back((ob_list));
		batch_prob_list.push_back((prob_list));
		batch_mu_list.push_back((mu_list));
		batch_sigma_list.push_back((sigma_list));
		batch_vpred_list.push_back((valuePred_list));
		batch_vtarg_list.push_back((valueTarg_list));
		batch_valueTargTD1.push_back((valueTargTD1));
		batch_traj_length.push_back(traj_length);
	}

	void standardizeBatchVector(std::vector<std::vector< ScalarType >>& batch_list)
	{
		double mean = 0;
		double var = 0;
		int count = 0;
		for(int i = 0; i < batch_list.size(); i++)
		{
			for (int j = 0; j < batch_traj_length[i]; j++)
			{
				mean += batch_list[i][j];
				count++;
			}
        }
		mean /= count;
		for (int i = 0; i < batch_list.size(); i++)
		{
			for (int j = 0; j < batch_traj_length[i]; j++)
			{
				var += (batch_list[i][j] - mean)*(batch_list[i][j] - mean);
			}
		}
		var /= count;
        if(var ==0){
            throw std::runtime_error("Cannot standardizeVector because var == 0.\n");
        }
        double std = sqrt(var);
		for (int i = 0; i < batch_list.size(); i++)
		{
			for (int j = 0; j < batch_traj_length[i]; j++)
			{
				batch_list[i][j] = (batch_list[i][j] - mean) / std;
			}
		}
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

	void save_tsv(std::vector<double> list, char* file_name) {
		std::ofstream file;
		file.open(file_name);
		for(int i = 0; i < list.size(); i++) {
			file << list[i] << "\t";
		}
		file.close();
	}

    ValueFuncWrapper valueFunc;
    //PolicyWrapper oldPolicy;
    std::vector<ScalarType> valuePred_list;
    std::vector<ScalarType> valueTarg_list;
	std::vector<ScalarType> valueTargTD1;
	int value_batch_size = 64;
	int update_batch_size = 64;
	int zero_derivatives = 0;
	int non_zero_derivatives = 0;

	int batch_backprop_count = 0;
	std::vector<int> shuffle_order;

    int m_timesteps_per_episode=0;
    double m_lambda = 0.98;
	double eps = 0.15;
	double best_return = 0;
	double best_return_iteration = 0;

    // lists of data over the full batch
    std::vector<std::vector<ScalarType>>                batch_adv_list;
    std::vector<std::vector<std::vector<ScalarType>>>   batch_ac_list;
    std::vector<std::vector<std::vector<ScalarType>>>   batch_ob_list;
    std::vector<std::vector<double>>                    batch_prob_list;
	std::vector<std::vector<std::vector<ScalarType>>>   batch_mu_list;
	std::vector<std::vector<std::vector<ScalarType>>>   batch_sigma_list;
    std::vector<std::vector<ScalarType>>                batch_vpred_list;
    std::vector<std::vector<ScalarType>>                batch_vtarg_list;
	std::vector<std::vector<ScalarType>>				batch_valueTargTD1;
	std::vector<int>									batch_traj_length;
	
	std::vector<double>									avg_returns_list;



};
