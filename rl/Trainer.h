#pragma once
#include <stdlib.h>
#include "Environment.h"

class Trainer
{
public:
    
protected:
    Trainer(Environment * new_env): env(new_env) {}
    Environment * env;
};
