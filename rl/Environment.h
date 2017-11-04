#pragma once
#include "NeuralNet.h"
#include "config.h"
#include <math.h>

class Environment
{
public:
    virtual void step(const std::vector<ScalarType>& actions) = 0;
    virtual ScalarType getReward() = 0;
    virtual const std::vector<ScalarType>& getState() = 0;
    virtual void reset() = 0;
    virtual int getActionSpaceDimensions() = 0;
    virtual int getStateSpaceDimensions() = 0;
	virtual bool earlyAbort() = 0; //if the trajectory should be stopped early

    std::vector<ScalarType> state;
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
    MathPuzzleEnv(){
        state.resize(2);
    }
    
    virtual void step(const std::vector<ScalarType>& actions)
    {
        checkActionDimensions(actions);
        dx= actions[0];
        x+=dx;
        dy= actions[1];
        y+=dy;
    };
    virtual ScalarType getReward(){
        return -(sqrt(pow(x-10,2)+pow(y+10,2)));
        // return x > 1 ? 1 : -1;
    };
    virtual const std::vector<ScalarType>& getState(){
        state[0] = x;
        state[1] = y;
        return state;
    };
	virtual bool earlyAbort() { return false; }//never early abort in this env 

    virtual void reset(){ x=1; y=1;}
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
    ScalarType dx=0;
    ScalarType y=1;
    ScalarType dy=0;
};
