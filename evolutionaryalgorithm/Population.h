#pragma once
#include <vector>
#include <memory>

template<typename Individual>
class Population 
{
public:
	Population() = default;
	~Population() = default;
	
	Population(const Population& copy_this) = delete;
	Population& operator=(const Population& copy_this) = delete;
	
	Population(Population&& move_this) = delete;
	Population& operator=(Population&& move_this) = delete;
	
private:
	std::vector<std::shared_ptr<Individual>> m_population;
};
