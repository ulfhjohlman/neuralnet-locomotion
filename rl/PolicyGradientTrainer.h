#pragma once
#include "Trainer.h"
#include "stdlib.h"
#include "PolicyWrapper.h"

class PolicyGradientTrainer : public Trainer
{
public:
    PolicyGradientTrainer(Environment * new_env, LayeredNeuralNet * new_policy_net)
            : Trainer(new_env),action_space_dim(env->getActionSpaceDimensions()),
            state_space_dim(env->getStateSpaceDimensions()),
            policy(new_policy_net,state_space_dim,action_space_dim) {
				if (normalize_obs) {
					for (int i = 0; i < state_space_dim; i++) {
						running_obs_mean.push_back(0);
						running_obs_m2.push_back(0);
						running_obs_var.push_back(1e-4);
					}
				}
            }

    void trainPG(int max_episodes, int timesteps_per_episode, int batch_size)
    {
        resizeLists(timesteps_per_episode);
        double mean_loss;
		double mean_loss_batch = 0;
		double mean_return_batch = 0;
        for(int i=0; i < max_episodes; i++)
        {
            //generate a trajectory (an episode)
			generateTrajectory(timesteps_per_episode);

            //at end of an episode
			{
				calcAdvFunc();
				standardizeVector(adv_list);
				mean_loss = calcLoss();
				mean_loss_batch += mean_loss;
				mean_return_batch += episode_return;
				//printf("Episode: %d. \tMean loss: %lf\n", i, mean_loss);

				calcGradients();


				backpropGradients();
				//cache loss gradients


				//if end of a minibatch
				if ((i + 1) % batch_size == 0)
				{
					mean_return_batch /= batch_size;
					mean_loss_batch = mean_loss_batch / static_cast<double>(batch_size);
					policy.popCacheLayerParams();
					policy.updateParams();
					//log progress
					if ((i / batch_size) % 10 == 0) {
						printf(" ---- Batch Update %d ---- \tmean return: %lf \n", i / batch_size, mean_return_batch);
					}
					//std::cout << "Loss pre optimization:\t" << preloss << " Post:\t" << postloss << "\n";

					mean_loss_batch = 0;
					mean_return_batch = 0;

				}

                //reset stuff
                env->reset();
                clearLists(); //technically not necessary
				resizeLists(timesteps_per_episode);
	
            }

        }
    }

protected:
    virtual void clearLists()
    {
        rew_list.clear();
        adv_list.clear();
        ac_list.clear();
        ob_list.clear();
        loss_list.clear();
        prob_list.clear();
		mu_grad_list.clear();
		sigma_grad_list.clear();
        mu_list.clear();
		sigma_list.clear();
    }

    virtual void resizeLists(int len)
    {
        rew_list.resize(len);
        adv_list.resize(len);
        ac_list.resize(len);
        ob_list.resize(len);
        loss_list.resize(len);
        prob_list.resize(len);
		mu_grad_list.resize(len);
		sigma_grad_list.resize(len);
		mu_list.resize(len);
		sigma_list.resize(len);
    }

private:
    virtual void calcAdvFunc()
    {
        ScalarType decayed_rew = 0;
        for(int i = traj_length-1; i >= 0; i--)
        {
            decayed_rew = rew_list[i] + decayed_rew * m_gamma;
            adv_list[i] = decayed_rew;
        }

    }

    // Loss with AdvFun estimate = decayed reward.
    // Returns mean loss over the episode
    virtual double calcLoss()
    {
        ScalarType meanLoss = 0;
        for(int i = 0; i < traj_length; i++)
        {
            loss_list[i] = - log(prob_list[i])* adv_list[i];
            meanLoss+=loss_list[i];
        }
        meanLoss /= ac_list.size();
        return meanLoss;
    }

    //assuming Gaussian mean outputs form network
    virtual void calcGradients()
    {
        //TODO:vectorize with eigen instead of std::vectors
        for(int i = 0 ;i < traj_length ;i++)
        {
            for(int j = 0 ; j < action_space_dim; j++ )
            {
				mu_grad_list[i][j] *= -adv_list[i]; //minus for ObjFunc -> LossFunc
				sigma_grad_list[i][j] *= -adv_list[i]; //TODO: speed up through eigen vectorisation instead of std::vectors
            }
        }
    }
protected:
    virtual void backpropGradients()
    {
        MatrixType x;
        MatrixType ob_input;
        for(int i =0; i<traj_length;i++)
        {
            //FORWARD PASS OBSERVATION i AGAIN BEFORE BACKPROPING ERROR GRADIENT i
            Eigen::Map<MatrixType> ob_input(ob_list[i].data(),state_space_dim,1);
            policy.input(ob_input);

			//stack the error gradients and backprop them
			std::vector<ScalarType> tmp; 
			tmp.insert(tmp.end(), mu_grad_list[i].begin(), mu_grad_list[i].end());//unnecessary coppying being performed :/ EIGENIZE later
			tmp.insert(tmp.end(), sigma_grad_list[i].begin(), sigma_grad_list[i].end());//unnecessary coppying being performed :/ EIGENIZE later
            Eigen::Map<MatrixType> x(tmp.data(),action_space_dim*2,1);
            policy.backprop(x);
            policy.cacheLayerParams();
        }
    }

    void standardizeVector(std::vector<ScalarType>& list)
    {
        double mean=0;
        double var=0;
        for(int i =0;i < list.size();i++)
        {
            mean += list[i];
        }
        mean /= list.size();
        for(int i =0;i < list.size();i++)
        {
            var += (list[i]-mean)*(list[i]-mean);
        }
        var /= list.size();
        if(var ==0){
            throw std::runtime_error("Cannot standardizeVector because var == 0.\n");
        }
        double std = sqrt(var);
        for(int i =0;i < list.size();i++)
        {
            list[i] = (list[i]-mean)/ std;
        }
    }

    virtual void generateTrajectory(int traj_max_length)
    {
        episode_return = 0;
		traj_length = traj_max_length;  //can be shortened if env->earlyAbort() flags true
        for(int i=0; i< traj_max_length;i++)
        {
            ob = env->getState();
			if (normalize_obs) { //normalization according to a running mean/var
				count_obs++; 
				for (int j = 0; j < ob.size(); j++)
				{					
					double delta = ob[j] - running_obs_mean[j];
					running_obs_mean[j] += delta / count_obs;
					double delta2 = ob[j] - running_obs_mean[j];
					running_obs_m2[j] += delta*delta2;
					if (count_obs != 1) {
						running_obs_var[j] = running_obs_m2[j] / (count_obs - 1);
					}
					ob[j] = (ob[j] - running_obs_mean[j]) / sqrt(running_obs_var[j]);
				}
			}
			/*


    for x in data:

        delta = x - mean
        mean += delta/n
        delta2 = x - mean
        M2 += delta*delta2

    if n < 2:
        return float('nan')
    else:
        return M2 / (n - 1)
		*/
            #ifdef _DEBUG
                if(ob.size() != state_space_dim){
                    char* str = new char[100];
                    sprintf(str,"Observation (%zd)!= state space size %d\n",ob.size(),state_space_dim);
                    throw std::runtime_error(str);
                }
            #endif

            ob_list[i] = ob;
			
            ac = policy.samplePolicy(ob);
			
            #ifdef _DEBUG
                if(ac.size() != action_space_dim){
                    char* str = new char[100];
                    sprintf(str,"Sampled policy ()%zd)!= action space size %d\n",ac.size(),action_space_dim);
                    throw std::runtime_error(str);
                }
            #endif
            ac_list[i] = ac;
			prob_list[i] = policy.getCumulativeProb();
            mu_list[i] = policy.getMu();
			sigma_list[i] = policy.getSigma();
            env->step(ac);

            rew_list[i] = env->getReward();
            episode_return += rew_list[i];

			mu_grad_list[i] = policy.getlocalMuObjectiveGradient(); // modified later by advFunc
			sigma_grad_list[i] = policy.getlocalSigmaObjectiveGradient(); // modified later by advFunc

			if (env->earlyAbort())
			{
				traj_length = i+1;
				break;
			}
		
        }
    }

    void print_state(const std::vector<ScalarType>& obs)
    {
        std::cout << "State: \t(";
        for(auto& ob: obs)
        {
            std::cout << ob << " ";
        }
        std::cout << ")\n";
    }


	std::vector<double> running_obs_var;
	std::vector<double> running_obs_mean;
	std::vector<double> running_obs_m2;
	long long count_obs = 0;

	bool normalize_obs =	false;
    ScalarType m_gamma = 0.995;
    int state_space_dim;
    int action_space_dim;
    PolicyWrapper policy;
	int traj_length;
    std::vector<ScalarType> ob;
    std::vector<ScalarType> ac;
    std::vector<ScalarType> adv_list;
    std::vector<ScalarType> rew_list;
    std::vector<ScalarType> loss_list;
    std::vector<double> prob_list;
	std::vector<std::vector<ScalarType>> mu_list;
	std::vector<std::vector<ScalarType>> sigma_list;
    std::vector<std::vector<ScalarType>> ob_list;
    std::vector<std::vector<ScalarType>> ac_list;
	std::vector<std::vector<ScalarType>> mu_grad_list;
	std::vector<std::vector<ScalarType>> sigma_grad_list;
	double episode_return;
};
