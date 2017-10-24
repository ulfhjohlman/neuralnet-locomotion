#pragma once
#include "PolicyGradientTrainer.h"

class PPOTrainer : public PolicyGradientTrainer
{
public:
    PPOTrainer(Environment * new_env, LayeredNeuralNet * new_policy, LayeredNeuralNet * new_valueFunc)
            : PolicyGradientTrainer(new_env, new_policy), valueFunc(new_valueFunc)  {}

private:
    LayeredNeuralNet * valueFunc;
};
