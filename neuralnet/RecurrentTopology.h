#pragma once
#include "LayeredTopology.h"

class RecurrentTopology  : public LayeredTopology
{
public:
	RecurrentTopology() = default;
	RecurrentTopology(std::initializer_list<int> baseLayerSize) :
		LayeredTopology(baseLayerSize), m_timeDelayLevels(0) { }
	virtual ~RecurrentTopology() = default;

	virtual void addRecurrentLayer() { }
	
	
private:
	std::vector<LayeredTopology> m_layers;
	int m_timeDelayLevels;
};
