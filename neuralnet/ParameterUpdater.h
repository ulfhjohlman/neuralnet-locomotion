#pragma once
#include<stdlib.h>
#include<config.h>
//Link this class to a set of parameters and their respective gradients
//Then call it to update them according to the chosen algorithm
class ParameterUpdater
{
public:
        ParameterUpdater() = delete;
        ParameterUpdater(ScalarType lr): m_learning_rate(lr) {}
        virtual ~ParameterUpdater() = default;

        void setLearningRate(ScalarType new_lr) { m_learning_rate = new_lr; }
        const ScalarType getLearningRate() { return m_learning_rate; }

        //vanila SGD update
        virtual void updateParameters()
        {
            checkParamsLinked();
            for(int i=0; i<params.size();i++)
            {
                params[i]->array() -= m_learning_rate * params_gradients[i]->array();
            }
        }

        //add a parameter and its gradient to those maintained and updated by this class
        virtual void addParam(MatrixType* param, MatrixType* param_gradient)
        {
            params.push_back(param);
            params_gradients.push_back(param_gradient);
        }

        virtual void linkLayerParams(Layer* layer)
        {
            addParam(&layer->m_weights, &layer->m_gradients_weights);
            addParam(&layer->m_bias, &layer->m_gradients_bias);
        }


protected:
        void checkParamsLinked()
        {
            #ifdef _DEBUG
                if(params.size() == 0 || params_gradients.size() == 0)
                    {throw std::runtime_error("ParamUpdater not linked to any params when updating :(\n");}
            #endif
        }
        std::vector<MatrixType*> params;
        std::vector<MatrixType*> params_gradients;
        ScalarType m_learning_rate;

        void destroyMatrixLists(std::vector<MatrixType*> list)
        {
            for(auto& ptr : list)
            {
                if(ptr)
                {
                    delete(ptr);
                }
            }
            list.clear();
        }
};


class MomentumUpdater : public ParameterUpdater
{
public:
    MomentumUpdater(ScalarType lr, ScalarType mu = 0.9) : ParameterUpdater(lr) , m_mu(mu) {}
    virtual ~MomentumUpdater()
    {
        destroyMatrixLists(m_momentum);
    }

    virtual void updateParameters()
    {
        checkParamsLinked();
        for(int i=0; i<params.size();i++)
        {
            m_momentum[i]->array() = m_mu*m_momentum[i]->array() - m_learning_rate * params_gradients[i]->array();
            params[i]->array() +=m_momentum[i]->array();

        }
    }

    virtual void addParam(MatrixType* param, MatrixType* param_gradient)
    {
        ParameterUpdater::addParam(param,param_gradient);
        MatrixType* x = new MatrixType();
        x->resizeLike(*param);
        x->setZero();
        m_momentum.push_back(x);
    }

    //for annealing, aka momentum decay
    void setMu(ScalarType new_mu) { m_mu = new_mu; }

protected:
    ScalarType m_mu;
    std::vector<MatrixType*> m_momentum;
};

class NesterovMomentumUpdater : public MomentumUpdater
{
public:
    NesterovMomentumUpdater(ScalarType lr, ScalarType mu = 0.9) : MomentumUpdater(lr,mu) {}
    virtual ~NesterovMomentumUpdater()
    {
        destroyMatrixLists(m_momentum_previous);
    }
    virtual void updateParameters()
    {
        checkParamsLinked();
        for(int i=0; i<params.size();i++)
        {
            m_momentum_previous[i] = m_momentum[i];
            m_momentum[i]->array() = m_mu*m_momentum[i]->array() - m_learning_rate * params_gradients[i]->array();
            params[i]->array() += -m_mu*m_momentum_previous[i]->array() + ((ScalarType)1 + m_mu)*m_momentum[i]->array();
        }
    }

    virtual void addParam(MatrixType* param, MatrixType* param_gradient)
    {
        MomentumUpdater::addParam(param,param_gradient);
        MatrixType * x = new MatrixType();
        x->resizeLike(*param);
        x->setZero();
        m_momentum_previous.push_back(x);
    }

private:
    std::vector<MatrixType*> m_momentum_previous;
};

class AdagradUpdater : public ParameterUpdater
{
public:
    //warning: can stop learning to early, especially in deep nets
    //typical epsilon values ~[1e-4,1e-8]
    AdagradUpdater(ScalarType lr, ScalarType epsilon = 1e-6) : ParameterUpdater(lr) , m_epsilon(epsilon) {}
    virtual ~AdagradUpdater()
    {
        destroyMatrixLists(m_cache);
    }

    virtual void updateParameters()
    {
        checkParamsLinked();
        for(int i=0; i<params.size();i++)
        {
            m_cache[i]->array() += params_gradients[i]->array().square();
            params[i]->array() += -m_learning_rate * params_gradients[i]->array() / (m_cache[i]->array().sqrt() + m_epsilon);
        }
    }

    virtual void addParam(MatrixType* param, MatrixType* param_gradient)
    {
        ParameterUpdater::addParam(param,param_gradient);
        MatrixType * x = new MatrixType();
        x->resizeLike(*param);
        x->setZero();
        m_cache.push_back(x);
    }

protected:
    std::vector<MatrixType*> m_cache;
    ScalarType m_epsilon;
};

class RMSPropUpdater : public AdagradUpdater
{ //aka leaky Adagrad, typical decay_rate ~[0.9,0.999]
public:
    RMSPropUpdater(ScalarType lr, ScalarType epsilon = 1e-6, ScalarType decay_rate = 0.99)
                        : AdagradUpdater(lr,epsilon) , m_decay_rate(decay_rate) {}

    virtual void updateParameters()
    {
        checkParamsLinked();
        for(int i=0; i<params.size();i++)
        {
            m_cache[i]->array() = m_decay_rate*m_cache[i]->array() + (1-m_decay_rate)*params_gradients[i]->array().square();
            params[i]->array() += -m_learning_rate * params_gradients[i]->array() / (m_cache[i]->array().sqrt() + m_epsilon);
        }
    }
private:
    ScalarType m_decay_rate;
};

class AdamUpdater : public ParameterUpdater
{ // typical hyperparameters ~ b1=0.9 b2=0.999 e=1e-8
public:
    AdamUpdater(ScalarType lr, ScalarType epsilon=1e-6, ScalarType b1=0.9, ScalarType b2=0.999) : ParameterUpdater(lr),
        m_epsilon(epsilon) , m_b1(b1), m_b2(b2) {}
    virtual ~AdamUpdater()
    {

        destroyMatrixLists(m_m);
        destroyMatrixLists(m_v);
    }

    virtual void updateParameters()
    {
        checkParamsLinked();
        MatrixType mt,vt;

        for(int i=0; i<params.size();i++)
        {
            m_m[i]->array() = m_b1 * m_m[i]->array() + ((ScalarType)1-m_b1)*params_gradients[i]->array();
            mt.array() = m_m[i]->array() / ((ScalarType)1 - pow(m_b1,t));

            m_v[i]->array() = m_b2 * m_v[i]->array() + ((ScalarType)1-m_b2)*(params_gradients[i]->array().square());
            vt.array() = m_v[i]->array() / ((ScalarType)1 - pow(m_b2,t));
            params[i]->array() += -m_learning_rate * mt.array() /(vt.array().sqrt() + m_epsilon);
        }
        t++;
    }
    virtual void addParam(MatrixType* param, MatrixType* param_gradient)
    {
        ParameterUpdater::addParam(param,param_gradient);
        MatrixType * x = new MatrixType();
        MatrixType * y = new MatrixType();
        x->resizeLike(*param);
        y->resizeLike(*param);
        x->setZero();
        y->setZero();
        m_m.push_back(x);
        m_v.push_back(y);
    }

private:
    int t = 1;
    ScalarType m_epsilon;
    ScalarType m_b1;
    ScalarType m_b2;
    std::vector<MatrixType*> m_v;
    std::vector<MatrixType*> m_m;
};
