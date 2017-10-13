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
            policy(PolicyWrapper(new_policy_net,state_space_dim,action_space_dim)) {
            }

    virtual void train(long max_iterations, long timesteps_per_run, long batch_size)
    {
        reward_list.reserve(batch_size * (timesteps_per_run+1));
        advantage_list.reserve(batch_size * (timesteps_per_run+1));

        for(int i=1; i < max_iterations; i++)
        {
            //if end of an episode
            if(i % timesteps_per_run == 0)
            {
                //calculate advFunc
                //calculate loss
                //calculate and cache loss gradients

                //if end of a minibatch
                if(i % (timesteps_per_run*batch_size) == 0)
                {
                    //pop cache
                    //backprop loss
                    //log progress

                }
                //reset stuff
                env->reset();
                reward_list.clear();
                advantage_list.clear();
            }
            //generate trajectories
            generateTrajectory(timesteps_per_run);

        }
    }
protected:
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

            ac = policy.samplePolicy(ob);

            #ifdef _DEBUG
                if(ac.size() != action_space_dim){
                    throw std::runtime_error("Sampled policy != action space size\n");
                }
            #endif
        }
    }



    int state_space_dim;
    int action_space_dim;
    PolicyWrapper policy;
    std::vector<ScalarType> ob;
    std::vector<ScalarType> ac;
    std::vector<ScalarType> advantage_list;
    std::vector<ScalarType> reward_list;
};
