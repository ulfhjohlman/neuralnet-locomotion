#pragma once
#include <vector>
#include <memory>

#include "Individual.h"

template<typename T>
class Population 
{
public:
	Population() = default;
	~Population() = default;

	//Generate a initial population
	virtual void initializePopulation() = 0;
	

private:
	std::vector<std::shared_ptr< Individual<T> >> m_population;

private:
	Population(const Population& copy_this) = delete;
	Population& operator=(const Population& copy_this) = delete;

	Population(Population&& move_this) = delete;
	Population& operator=(Population&& move_this) = delete;

};
