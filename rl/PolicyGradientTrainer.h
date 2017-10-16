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

    virtual void train(int max_episodes, int timesteps_per_run, int batch_size)
    {
        reserveLists(timesteps_per_run);
        ScalarType mean_loss;
        for(int i=1; i <= max_episodes; i++)
        {
            //generate a trajectory (an episode)
            generateTrajectory(timesteps_per_run);

            //at end of an episode
            {
                calcAdvFunc();
                mean_loss = calcLoss();
                calcGradients();

                backpropGradients(); //ASSUMING INPUTS OF RESPECTIVE FORWARD PASS :'(
                //cache loss gradients

                //if end of a minibatch
                if(i % batch_size == 0)
                {
                    //pop cache
                    //backprop loss
                    //log progress

                }
                //printf("Episode: %d. \tMean loss: %f\n", i,mean_loss);

                //reset stuff
                env->reset();
                clearLists();
            }

        }
    }
protected:
    void clearLists()
    {
        rew_list.clear();
        adv_list.clear();
        ac_list.clear();
        ob_list.clear();
        loss_list.clear();
        prob_list.clear();
        grad_list.clear();
    }

    void reserveLists(int len)
    {
        rew_list.reserve(len);
        adv_list.reserve(len);
        ac_list.reserve(len);
        ob_list.reserve(len);
        loss_list.reserve(len);
        prob_list.reserve(len);
        grad_list.reserve(len);
    }
    void calcAdvFunc()
    {
        ScalarType decayed_rew = 0;
        for(int i = ac_list.size()-1; i >= 0; i--)
        {
            decayed_rew = rew_list[i] + decayed_rew * rew_decay_rate;
            adv_list[i] = decayed_rew;
        }
    }

    // Loss with AdvFun estimate = decayed reward.
    // Returns mean loss over the episode
    ScalarType calcLoss()
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
    void calcGradients()
    {
        //TODO:vectorize with eigen instead of std::vectors
        for(int i = 0 ;i < grad_list.size();i++)
        {
            for(int j = 0 ; j < action_space_dim; j++ )
            {
                grad_list[i][j] *= adv_list[i];
            }
        }
    }

    void backpropGradients()
    {
        MatrixType x;
        MatrixType ob_input;
        for(int i =0; i<grad_list.size();i++)
        {
            //FORWARD PASS OBSERVATION i AGAIN BEFORE EACH BACKPROP GRAD
            Eigen::Map<MatrixType> ob_input(ob_list[i].data(),state_space_dim,1);
            policy.input(ob_input);

            Eigen::Map<MatrixType> x(grad_list[i].data(),action_space_dim,1);
            policy.backprop(x);
            policy.cacheLayerParams();
        }
    }

    void generateTrajectory(int traj_length)
    {
        for(int i=0; i< traj_length;i++)
        {

            ob = env->getState();

            #ifdef _DEBUG
                if(ob.size() != state_space_dim){
                    char* str = new char[100];
                    sprintf(str,"Observation (%ld)!= state space size %d\n",ob.size(),state_space_dim);
                    throw std::runtime_error(str);
                }
            #endif

            ob_list.push_back(ob);
            rew_list.push_back(env->getReward());
            ac = policy.samplePolicy(ob);

            #ifdef _DEBUG
                if(ac.size() != action_space_dim){
                    char* str = new char[100];
                    sprintf(str,"Sampled policy ()%ld)!= action space size %d\n",ac.size(),action_space_dim);
                    throw std::runtime_error(str);
                }
            #endif
            ac_list.push_back(ac);
            env->step(ac);

            grad_list.push_back(policy.getLocalErrorGradient()); // modified later by advFunc

        }
    }


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
};
