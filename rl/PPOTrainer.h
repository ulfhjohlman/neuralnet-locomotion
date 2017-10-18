#pragma once
#include "PolicyGradientTrainer.h"
#include "ValueFuncWrapper.h"

class PPOTrainer : public PolicyGradientTrainer
{
public:
    PPOTrainer(Environment * new_env, LayeredNeuralNet * new_policy, LayeredNeuralNet * new_valueFunc)
            : PolicyGradientTrainer(new_env, new_policy), valueFunc(new_valueFunc,state_space_dim)  {}

    void set_GAE_lambda(double new_lambda)
    {
        m_lambda = new_lambda;
    }
    void set_GAE_gamma(double new_gamma)
    {
        m_gamma = new_gamma;
    }

protected:
    virtual void resizeLists(int len)
    {
        PolicyGradientTrainer::resizeLists(len);
        valuePred_list.resize(len);
    }
    virtual void clearLists()
    {
        PolicyGradientTrainer::clearLists();
        valuePred_list.clear();
    }

    //produces the value prediction lists using the valueFunction netowk
    void makeValuePredictions()
    {
        for(int i = 0 ; i<ob_list.size() ; i++)
        {
            Eigen::Map<MatrixType> ob_matrix(ob_list[i].data(),state_space_dim,1);
            valuePred_list[i] = valueFunc.predict(ob_matrix)(0,0);
        }
    }

    //produce generalized advantage estimates using value network and state list
    void GAE(){
        makeValuePredictions();
        double delta = rew_list.back() - valuePred_list.back();
        adv_list.back() = delta;

        for(int i=adv_list.size()-2;i>=0;i--)
        {
            delta = rew_list[i] + m_gamma * valuePred_list[i+1] - valuePred_list[i];
            adv_list[i] = delta + m_lambda*m_gamma*adv_list[i+1];
        }
    }

    ValueFuncWrapper valueFunc;
    std::vector<ScalarType> valuePred_list;
    double m_lambda;
    double m_gamma;
};
