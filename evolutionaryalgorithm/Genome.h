#pragma once
#include <vector>
#include <iostream>

//subclass for specific encoding
class Genome 
{
public:
	Genome() = default;
	virtual ~Genome() = default;

	Genome& operator=(const Genome& copy_this) {
		std::cout << "genome copy" << std::endl;
		return *this;
	}

	virtual void mutate(float mutation_probability) = 0;
	//virtual std::vector<Genome*> cut(int cuts) = 0;
private:
	
};