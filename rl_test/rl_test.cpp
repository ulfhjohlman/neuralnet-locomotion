#include "Trainer.h"
#include "PPOTrainer.h"
#include "PolicyGradientTrainer.h"
#include <stdlib.h>
#include <iostream>
#include "config.h"
#include "LayeredTopology.h"
#include "LayeredNeuralNet.h"
#include "CascadeNeuralNet.h"
#include "ParameterUpdater.h"
int ppo_test()
{
    std::cout << "Starting ppo_test\n";

    //initalize environment
    MathPuzzleEnv env;
    int action_space_dim = env.getActionSpaceDimensions();
    int state_space_dim = env.getStateSpaceDimensions();

    //constructing networks
    std::vector<int> layerSizesPolicy {state_space_dim,4,4,action_space_dim};
    std::vector<int> layerSizesOldPolicy {state_space_dim,4,4,action_space_dim};
    std::vector<int> layerSizesValueFunc {state_space_dim,4,4,1};
    const int relu = Layer::LayerType::relu;
    const int inputLayer = Layer::LayerType::inputLayer;
    const int noactiv = Layer::noActivation;
    const int softmax = Layer::LayerType::softmax;
    const int sigmoid = Layer::LayerType::sigmoid;
    const int tanh = Layer::LayerType::tanh;
    std::vector<int> layerTypesPolicy {inputLayer, tanh,tanh,noactiv};
    std::vector<int> layerTypesOldPolicy {inputLayer, tanh,tanh,noactiv};
    std::vector<int> layerTypesValueFunc {inputLayer, tanh,tanh,noactiv};

    //networks gain ownership/claening responsibilities for these topologies
    LayeredTopology* topPolicy = new LayeredTopology(layerSizesPolicy,layerTypesPolicy);
    LayeredTopology* topOldPolicy = new LayeredTopology(layerSizesOldPolicy,layerTypesOldPolicy);
    LayeredTopology* topValueFunc = new LayeredTopology(layerSizesValueFunc,layerTypesValueFunc);

    LayeredNeuralNet policy(topPolicy);
    LayeredNeuralNet oldPolicy(topOldPolicy);
    LayeredNeuralNet valueFunc(topValueFunc);

    policy.initializeXavier();
    oldPolicy.initializeXavier();
    valueFunc.initializeXavier();

    //ParameterUpdater
    AdamUpdater policyUpdater(1e-3);
    AdamUpdater oldPolicyUpdater(1e-3);
    // ParameterUpdater policyUpdater(1e-5);
    // RMSPropUpdater policyUpdater(1e-3,1e-8,0.99);
    AdamUpdater valueFuncUpdater(1e-3);

    policy.setParameterUpdater(policyUpdater);
    oldPolicy.setParameterUpdater(oldPolicyUpdater);
    valueFunc.setParameterUpdater(valueFuncUpdater);

    //set up training algorithm
    // PolicyGradientTrainer trainer(&env,&policy);
    PPOTrainer trainer(&env,&policy,&oldPolicy,&valueFunc);
    //arguments: iterations,  batchsize, timesteps_episode, minibatch_size, epochs
    trainer.trainPPO(1000,1,64,1,1);


    return 0;
}
int pg_test()
{
    std::cout << "Starting pg_test\n";

    //initalize environment
    MathPuzzleEnv env;
    int action_space_dim = env.getActionSpaceDimensions();
    int state_space_dim = env.getStateSpaceDimensions();

    //constructing networks
    std::vector<int> layerSizesPolicy {state_space_dim,4,4,action_space_dim};
    const int relu = Layer::LayerType::relu;
    const int inputLayer = Layer::LayerType::inputLayer;
    const int noactiv = Layer::noActivation;
    const int softmax = Layer::LayerType::softmax;
    const int sigmoid = Layer::LayerType::sigmoid;
    const int tanh = Layer::LayerType::tanh;
    std::vector<int> layerTypesPolicy {inputLayer, tanh,tanh,noactiv};

    //networks gain ownership/claening responsibilities for these topologies
    LayeredTopology* topPolicy = new LayeredTopology(layerSizesPolicy,layerTypesPolicy);

    LayeredNeuralNet policy(topPolicy);

    policy.initializeXavier();

    //ParameterUpdater
    AdamUpdater policyUpdater(1e-3);

    policy.setParameterUpdater(policyUpdater);

    //set up training algorithm
    PolicyGradientTrainer trainer(&env,&policy);
    //arguments: max_episodes, timesteps_per_episode, batch_size
    trainer.trainPG(50024,16,4);


    return 0;
}

int main()
{
    std::cout << " ============ Running rl_test... ============ \n";
	#ifdef _DEBUG
			std::cout << "_DEBUG FLAG ON\n";
	#else
			std::cout << "_DEBUG FLAG OFF\n";
    #endif
    try{
        ppo_test();
        // pg_test();
    }
    catch(const std::runtime_error& e)
    {
        std::cout << "ERROR: \n" << e.what();
    }
    std::cout << " ============== rl_test ended ============== \n";
    system("pause");
    return 0;
}
