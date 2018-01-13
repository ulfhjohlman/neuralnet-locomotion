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
#include "RL_MJ_Environment.h"
int ppo_test()
{
    std::cout << "Starting ppo_test\n";

    //initalize environment
    MathPuzzleEnv env;
    int action_space_dim = env.getActionSpaceDimensions();
    int state_space_dim = env.getStateSpaceDimensions();

    //constructing networks
    std::vector<int> layerSizesPolicy {state_space_dim,16,16,action_space_dim};
    std::vector<int> layerSizesValueFunc {state_space_dim,16,16,1};
    const int relu = Layer::LayerType::relu;
    const int inputLayer = Layer::LayerType::inputLayer;
    const int noactiv = Layer::noActivation;
    const int softmax = Layer::LayerType::softmax;
    const int sigmoid = Layer::LayerType::sigmoid;
    const int tanh = Layer::LayerType::tanh;
    std::vector<int> layerTypesPolicy {inputLayer, tanh,tanh,tanh};
    std::vector<int> layerTypesValueFunc {inputLayer, tanh,tanh,tanh};

    //networks gain ownership/claening responsibilities for these topologies
    LayeredTopology* topPolicy = new LayeredTopology(layerSizesPolicy,layerTypesPolicy);
    LayeredTopology* topValueFunc = new LayeredTopology(layerSizesValueFunc,layerTypesValueFunc);

    LayeredNeuralNet policy(topPolicy);
    LayeredNeuralNet valueFunc(topValueFunc);

    policy.initializeXavier();
    valueFunc.initializeXavier();

    //ParameterUpdater
    AdamUpdater policyUpdater(1e-4);
    // ParameterUpdater policyUpdater(1e-5);
    // RMSPropUpdater policyUpdater(1e-3,1e-8,0.99);
     AdamUpdater valueFuncUpdater(1e-4);
	//ParameterUpdater valueFuncUpdater(1e-3);

    policy.setParameterUpdater(policyUpdater);
    valueFunc.setParameterUpdater(valueFuncUpdater);

    //set up training algorithm
    // PolicyGradientTrainer trainer(&env,&policy);
    PPOTrainer trainer(&env,&policy,&valueFunc);
    //arguments: iterations,  batchsize, timesteps_episode, minibatch_size, epochs

    trainer.trainPPO(10000,16,12,4,2,0);


    return 0;
}
int ppo_mj_test()
{
	std::cout << "Starting ppo_mj_test\n";

	//initalize environment
	//InvDoublePendEnv env;
	//HumanoidEnv env;
	//HumanoidEnv2 env;
	Walker2dEnv env;
	//HopperEnv env;
	//AntEnv env;
	int action_space_dim = env.getActionSpaceDimensions();
	int state_space_dim = env.getStateSpaceDimensions();

	//constructing networks
	/*
	int layer1size = 10 * state_space_dim;
	int layer3size = 10 * action_space_dim;
	int geo_mean1 = static_cast<int>(sqrt(layer1size*layer1size + layer3size*layer3size)); //geometric mean of layer 1 and 3 for layer 2
	int geo_mean2 = static_cast<int>(sqrt(layer1size*layer1size + 5*5)); 
	std::vector<int> layerSizesPolicy{ state_space_dim,state_space_dim*10,geo_mean1,action_space_dim*10,action_space_dim };
	std::vector<int> layerSizesValueFunc{ state_space_dim,state_space_dim*10,geo_mean2,5,1 };
	*/
	std::vector<int> layerSizesPolicy{ state_space_dim,300,200,100,action_space_dim };
	std::vector<int> layerSizesValueFunc{ state_space_dim,300,100,20,1 };
	const int relu = Layer::LayerType::relu;
	const int inputLayer = Layer::LayerType::inputLayer;
	const int noactiv = Layer::noActivation;
	const int softmax = Layer::LayerType::softmax;
	const int sigmoid = Layer::LayerType::sigmoid;
	const int tanh = Layer::LayerType::tanh;
	std::vector<int> layerTypesPolicy{ inputLayer, tanh,tanh,tanh,noactiv};
	std::vector<int> layerTypesValueFunc{ inputLayer, tanh,tanh,tanh,noactiv };


	//networks gain ownership/claening responsibilities for these topologies
	LayeredTopology* topPolicy = new LayeredTopology(layerSizesPolicy, layerTypesPolicy);
	LayeredTopology* topValueFunc = new LayeredTopology(layerSizesValueFunc, layerTypesValueFunc);

	LayeredNeuralNet policy(topPolicy);
	LayeredNeuralNet valueFunc(topValueFunc);

	policy.initializeXavier();
	valueFunc.initializeXavier();

	

	//ParameterUpdater
	AdamUpdater policyUpdater(1e-4);
	AdamUpdater valueFuncUpdater(1e-4);

	policy.setParameterUpdater(policyUpdater);
	valueFunc.setParameterUpdater(valueFuncUpdater);

	//set up training algorithm
	// PolicyGradientTrainer trainer(&env,&policy);
	PPOTrainer trainer(&env, &policy, &valueFunc);
	//arguments: iterations,  batchsize, timesteps_episode, minibatch_size, epochs
	int frameskip = 1;

	env.set_frameskip(frameskip);
	trainer.setName("ppo_mj_test");
	trainer.trainPPO(10000 , 32, 4096/frameskip, 64, 1,0);



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
    std::vector<int> layerSizesPolicy {state_space_dim,8,8,action_space_dim};
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
    trainer.trainPG(50024,10,4);


    return 0;
}
int pg_mj_test()
{
	std::cout << "Starting pg_test\n";

	//initalize environment
	HopperEnv env;
	int action_space_dim = env.getActionSpaceDimensions();
	int state_space_dim = env.getStateSpaceDimensions();

	//constructing networks
	std::vector<int> layerSizesPolicy{ state_space_dim,32,32,action_space_dim };
	const int relu = Layer::LayerType::relu;
	const int inputLayer = Layer::LayerType::inputLayer;
	const int noactiv = Layer::noActivation;
	const int softmax = Layer::LayerType::softmax;
	const int sigmoid = Layer::LayerType::sigmoid;
	const int tanh = Layer::LayerType::tanh;
	std::vector<int> layerTypesPolicy{ inputLayer, tanh,tanh,tanh };

	//networks gain ownership/claening responsibilities for these topologies
	LayeredTopology* topPolicy = new LayeredTopology(layerSizesPolicy, layerTypesPolicy);

	LayeredNeuralNet policy(topPolicy);

	policy.initializeXavier();

	//ParameterUpdater
	AdamUpdater policyUpdater(1e-4);

	policy.setParameterUpdater(policyUpdater);

	//set up training algorithm
	PolicyGradientTrainer trainer(&env, &policy);
	//arguments: max_episodes, timesteps_per_episode, batch_size

	int frameskip = 3;
	env.set_frameskip(frameskip);
	trainer.trainPG(1e10, 2048/frameskip, 8);


	return 0;
}

int ppo_with_initial_net(Environment * env, LayeredNeuralNet* policy_nn_ptr, LayeredNeuralNet* value_nn_ptr, int iterations, int max_time)
{
	std::cout << "Starting ppo_with_inital_net\n";
	//ParameterUpdater
	AdamUpdater policyUpdater(1e-4);
	AdamUpdater valueFuncUpdater(1e-4);

	policy_nn_ptr->setParameterUpdater(policyUpdater);
	value_nn_ptr->setParameterUpdater(valueFuncUpdater);

	//set up training algorithm
	// PolicyGradientTrainer trainer(&env,&policy);
	PPOTrainer trainer(env, policy_nn_ptr, value_nn_ptr);
	//arguments: iterations,  batchsize, timesteps_episode, minibatch_size, epochs

	trainer.setName("ppo_mj_test");
	trainer.trainPPO(iterations, 32, max_time, 64, 1, 1000);

	return 0;
}

void test_ppo_with_init_net() {
	Walker2dEnv env;
	LayeredNeuralNet policynet;
	policynet.load("best_net/ppo_mj_test");

	int action_space_dim = env.getActionSpaceDimensions();
	int state_space_dim = env.getStateSpaceDimensions();

	std::vector<int> layerSizesValueFunc{ state_space_dim,128,64,32,1 };
	const int inputLayer = Layer::LayerType::inputLayer;
	const int noactiv = Layer::noActivation;
	const int tanh = Layer::LayerType::tanh;
	std::vector<int> layerTypesValueFunc{ inputLayer, tanh,tanh,tanh,noactiv };

	//networks gain ownership/claening responsibilities for these topologies
	LayeredTopology* topValueFunc = new LayeredTopology(layerSizesValueFunc, layerTypesValueFunc);
	LayeredNeuralNet valuenet(topValueFunc);

	ppo_with_initial_net(&env, &policynet, &valuenet, 5000, 2000);
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
         ppo_mj_test();
		// pg_mj_test();
		//ppo_test();
         //pg_test();
		 test_ppo_with_init_net();
    }
	catch(const std::runtime_error& e)
    {
        std::cout << "ERROR: \n" << e.what();
    }
    std::cout << " ============== rl_test ended ============== \n";
    system("pause");
    return 0;
}
