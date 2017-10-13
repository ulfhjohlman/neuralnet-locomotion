#pragma once
#include "NeuralNet.h"
#include "config.h"
#include <math.h>

class Environment
{
public:
    virtual void step(const std::vector<ScalarType>& actions) = 0;
    virtual ScalarType getReward() = 0;
    virtual const std::vector<ScalarType> getState() = 0;
    virtual void reset() = 0;
    virtual int getActionSpaceDimensions() = 0;
    virtual int getStateSpaceDimensions() = 0;
};

/*
test class solving the problem of finding coordinates (x,y) = (10,-10).
State space: (x,y)
Action space: movement (dx ,dy)
reward: r = - (||x-10|| + ||y + 10||)
*/
class MathPuzzleEnv : public Environment
{
public:
    MathPuzzleEnv() = default;
    virtual void step(const std::vector<ScalarType>& actions)
    {
        checkActionDimensions(actions);
        x += actions[0];
        y += actions[1];
    };
    virtual ScalarType getReward(){
        return (sqrt(pow(x-10,2)) + sqrt(pow(y+10,2)));
    };
    virtual const std::vector<ScalarType> getState(){
        return std::vector<ScalarType>{x,y};
    };

    virtual void reset(){ x=0; y=0;}
    virtual int getActionSpaceDimensions(){return 2;}
    virtual int getStateSpaceDimensions(){return 2;}

    void checkActionDimensions(std::vector<ScalarType> actions){
        #ifdef _DEBUG
            if(getActionSpaceDimensions() != actions.size())
            {
                throw std::runtime_error("Actionspace dim incorrect\n");
            }
        #endif
    }
private:
    ScalarType x=1;
    ScalarType y=1;
};