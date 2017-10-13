#pragma once
#include "Trainer.h"
#include "stdlib.h"
#include "PolicyWrapper.h"

class PolicyGradientTrainer : public Trainer
{
public:
    PolicyGradientTrainer(Environment * new_env, NeuralNet * new_policy_net)
            : Trainer(new_env),action_space_dim(env->getActionSpaceDimensions()),
            state_space_dim(env->getStateSpaceDimensions()),
            policy(new_policy_net,state_space_dim,action_space_dim) {
            }

    virtual void train(long max_episodes, long timesteps_per_run, long batch_size)
    {
        rew_list.reserve(batch_size * (timesteps_per_run+1));
        adv_list.reserve(batch_size * (timesteps_per_run+1));
        ac_list.reserve(batch_size * (timesteps_per_run+1));
        ob_list.reserve(batch_size * (timesteps_per_run+1));
        loss_list.reserve(batch_size * (timesteps_per_run+1));

        for(int i=1; i < max_episodes; i++)
        {
            //generate a trajectory (an episode)
            generateTrajectory(timesteps_per_run);

            //at end of an episode
            {
                //calculate advFunc
                calcAdvFunc();
                calcLoss();
                //calculate and cache loss gradients

                //if end of a minibatch
                if(i % batch_size == 0)
                {
                    //pop cache
                    //backprop loss
                    //log progress

                }
                //reset stuff
                env->reset();
                rew_list.clear();
                adv_list.clear();
                ac_list.clear();
                ob_list.clear();
            }

        }
    }
protected:
    void calcAdvFunc()
    {
        ScalarType decayed_rew = 0;
        for(int i = ac_list.size()-1; i >= 0; i--)
        {
            decayed_rew = rew_list[i] + decayed_rew * rew_decay_rate;
            adv_list[i] = decayed_rew;
        }
    }

    void calcLoss()
    {
        
    }

    void generateTrajectory(long traj_length)
    {
        for(long i=0; i< traj_length;i++)
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
    std::vector<std::vector<ScalarType>> ob_list;
    std::vector<std::vector<ScalarType>> ac_list;
};
