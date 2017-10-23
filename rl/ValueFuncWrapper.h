#pragma once

#include "LayeredNeuralNet.h"
#include <stdexcept>

class ValueFuncWrapper
{
public:
    ValueFuncWrapper(LayeredNeuralNet * nn, int state_space_dim) : m_nn(nn){
        if(m_nn->getTopology()->getLayerSize(m_nn->getTopology()->getNumberOfLayers()-1)!= 1)
        {
            throw std::invalid_argument("Value Network should only have 1 output!\n");
        };
        if(m_nn->getTopology()->getLayerSize(0)!= state_space_dim)
        {
            throw std::invalid_argument("Value Network has wrong number of inputs!\n");
        };
        if(!m_nn->hasParameterUpdater())
        {
            throw std::invalid_argument("Value Network initialized without a ParameterUpdater\n");
        }
    }

    const MatrixType& predict(const MatrixType ob)
    {
        m_nn->input(ob);
        return m_nn->output();
    }

    void backprop(const MatrixType& error_grad)
    {
        m_nn->backprop(error_grad);
    }
    void cacheLayerParams()
    {
        m_nn->cacheLayerParams();
    }

    void popCacheLayerParams()
    {
        m_nn->popCacheLayerParams();
    }
    void updateParams()
    {
        m_nn->updateParameters();
    }

private:
    LayeredNeuralNet * m_nn;
    int state_space_dim;
};
