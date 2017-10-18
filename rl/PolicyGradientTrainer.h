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
            generateTrajectory(timesteps_per_episode);

            //at end of an episode
            {
                calcAdvFunc();
                mean_loss = calcLoss();
				mean_loss_batch += mean_loss;
				mean_return_batch += episode_return;
				//printf("Episode: %d. \tMean loss: %lf\n", i, mean_loss);
                calcGradients();

                backpropGradients();
                //cache loss gradients

                //if end of a minibatch
                if(i % batch_size == 0 && i>0)
                {
					mean_loss_batch = mean_loss_batch/ static_cast<double>(batch_size);
					policy.popCacheLayerParams();
				    policy.updateParams();
                    //log progress
					printf(" ---- Batch Update %d ---- \tmean return: %lf \tmean loss over batch: %lf \n", i / batch_size, mean_return_batch, mean_loss_batch);
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
    virtual void clearLists()
    {
        rew_list.clear();
        adv_list.clear();
        ac_list.clear();
        ob_list.clear();
        loss_list.clear();
        prob_list.clear();
        grad_list.clear();
    }

    virtual void resizeLists(int len)
    {
        rew_list.resize(len);
        adv_list.resize(len);
        ac_list.resize(len);
        ob_list.resize(len);
        loss_list.resize(len);
        prob_list.resize(len);
        grad_list.resize(len);
    }
    virtual void calcAdvFunc()
    {
        ScalarType decayed_rew = 0;
        for(int i = ac_list.size()-1; i >= 0; i--)
        {
            decayed_rew = rew_list[i] + decayed_rew * rew_decay_rate;
            adv_list[i] = decayed_rew;
        }
        standardizeVector(adv_list);
    }

    // Loss with AdvFun estimate = decayed reward.
    // Returns mean loss over the episode
    virtual ScalarType calcLoss()
    {
        ScalarType meanLoss = 0;
        for(int i = 0; i < ac_list.size(); i++)
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
        for(int i = 0 ;i < grad_list.size();i++)
        {
            for(int j = 0 ; j < action_space_dim; j++ )
            {
                grad_list[i][j] *= -adv_list[i];
            }
        }
    }

    virtual void backpropGradients()
    {
        MatrixType x;
        MatrixType ob_input;
        for(int i =0; i<grad_list.size();i++)
        {
            //FORWARD PASS OBSERVATION i AGAIN BEFORE BACKPROPING ERROR GRADIENT i
            Eigen::Map<MatrixType> ob_input(ob_list[i].data(),state_space_dim,1);
            policy.input(ob_input);
            Eigen::Map<MatrixType> x(grad_list[i].data(),action_space_dim,1);
            policy.backprop(x);
            policy.cacheLayerParams();
        }
    }

    virtual void generateTrajectory(int traj_length)
    {
        for(int i=0; i< traj_length;i++)
        {
			episode_return = 0;
            ob = env->getState();
            if( i == (traj_length-1) && print_ob_final )
            {
                print_state(ob);
            }
            #ifdef _DEBUG
                if(ob.size() != state_space_dim){
                    char* str = new char[100];
                    sprintf(str,"Observation (%zd)!= state space size %d\n",ob.size(),state_space_dim);
                    throw std::runtime_error(str);
                }
            #endif

            ob_list[i] = ob;
            rew_list[i] = env->getReward();
			episode_return += rew_list[i];
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
            env->step(ac);

            grad_list[i] = policy.getlocalObjectiveGradient(); // modified later by advFunc

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
        double std = sqrt(var);
        for(int i =0;i < list.size();i++)
        {
            list[i] = (list[i]-mean)/ std;
        }
    }


    bool print_ob_final = true;
    ScalarType rew_decay_rate = 0.99;
    int state_space_dim;
    int action_space_dim;
    PolicyWrapper policy;
    std::vector<ScalarType> ob;
    std::vector<ScalarType> ac;
    std::vector<ScalarType> adv_list;
    std::vector<ScalarType> rew_list;
    std::vector<ScalarType> loss_list;
    std::vector<ScalarType> prob_list;
    std::vector<std::vector<ScalarType>> ob_list;
    std::vector<std::vector<ScalarType>> ac_list;
    std::vector<std::vector<ScalarType>> grad_list;
	double episode_return;
};
