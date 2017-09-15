#pragma once
#include "LayeredTopology.h"
#include <algorithm>

class RecurrentTopology  : public LayeredTopology
{
public:
	RecurrentTopology() {
		m_topologies.push_back(this);
		m_topologyConnections.push_back(std::vector<int>());
	}
	RecurrentTopology(std::initializer_list<int> baseLayerSize) :
		LayeredTopology(baseLayerSize), m_timeDelayLevels(0) { }
	virtual ~RecurrentTopology() = default;

	virtual void addRecurrentLayeredTopology( LayeredTopology* topology, 
		int fromTopology, const std::vector<int>& fromLayers, 
		int toTopology, const std::vector<int>& toLayer ) {
		checkLayeredTopology(topology);
		m_topologies.push_back(topology);
	}
	
protected:
	inline void checkConnections(int fromTopology, const std::vector<int>& fromLayers,
		int toTopology, const std::vector<int>& toLayers) {
#ifdef _NEURALNET_DEBUG
		bool bfromTopology = fromTopology > -1 && fromTopology < m_topologies.size();
		bool btoTopology = toTopology > -1 && toTopology < m_topologies.size();
		if (!bfromTopology || !btoTopology)
			throw std::invalid_argument("Connection between topologies not possible.");

		auto validRange = [](const std::vector<int>& vec, int maxIndex) 
		{ return std::all_of(vec.begin(), vec.end(), [&maxIndex](int i) { return i < maxIndex; }); };

		size_t fromSize = m_topologies.back()->getNumberOfLayers();
		bool bfromLayers = fromLayers.size() > -1 && validRange(fromLayers, fromSize);

		size_t toSize = m_topologies[toTopology]->getNumberOfLayers();
		bool btoLayers = toLayers.size() > -1 && validRange(toLayers, toSize);

		if(!bfromLayers || !btoLayers)
			throw std::invalid_argument("Connecting layers not possible.");
#endif // _NEURALNET_DEBUG
	}

	inline void checkLayeredTopology(LayeredTopology* topology) {
#ifdef _NEURALNET_DEBUG
		if (topology == nullptr)
			throw std::invalid_argument("topology nullptr");
		if (topology->getNumberOfLayers() < 1)
			throw std::invalid_argument("No valid topology");
#endif // _NEURALNET_DEBUG
	}
	
private:
	std::vector<LayeredTopology*> m_topologies;
	std::vector<std::vector<int>> m_topologyConnections;
	int m_timeDelayLevels;
};
