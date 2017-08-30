#pragma once
#include <sstream>

template<typename Population, typename Individual>
class EvolutionStrategy 
{
public:
	EvolutionStrategy() = default;
	~EvolutionStrategy() = default;
	
	EvolutionStrategy(const EvolutionStrategy& copy_this) = delete;
	EvolutionStrategy& operator=(const EvolutionStrategy& copy_this) = delete;
	
	EvolutionStrategy(EvolutionStrategy&& move_this) = delete;
	EvolutionStrategy& operator=(EvolutionStrategy&& move_this) = delete;
	
	template< typename ...Operator >
	void evolve(Population& population);

	std::string statusLog() const;
	void clearLog();
protected:
	std::ostringstream m_log;
private:
	
};