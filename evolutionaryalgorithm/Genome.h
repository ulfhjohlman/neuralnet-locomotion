#pragma once
#include <vector>

//subclass for specific encoding
class Genome 
{
public:
	Genome() = default;
	virtual~Genome() = default;

	virtual void mutate(float mutation_probability) = 0;
	virtual std::vector<Genome*> cut(int cuts) = 0;
private:
	
};