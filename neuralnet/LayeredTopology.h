#pragma once
#include "Topology.h"
#include <vector>
#include <stdexcept>
#include <initializer_list>


class LayeredTopology : private Topology
{
public:
	LayeredTopology() = default;
	LayeredTopology(std::vector<int>& layerSize) : m_layerSizes(layerSize) { }
	LayeredTopology(std::initializer_list<int> layerSize) : m_layerSizes(layerSize) { }
	virtual ~LayeredTopology() = default;
	
	virtual void reserveLayers(int numberOfLayers) {
		m_layerSizes.reserve(numberOfLayers);
	}
	virtual void addLayer(int size) {
		checkSize(size);
		m_layerSizes.push_back(size);
	}
	virtual void insert(int index, int size) {
		checkSize(size);
		checkIndex(index);
		m_layerSizes.emplace(m_layerSizes.begin(), size);
	}
	virtual void removeLayer(int index) {
		checkIndex(index);
		m_layerSizes.erase(m_layerSizes.begin()+index);
	}
	virtual void resizeLayer(int index, int size) {
		checkSize(size);
		checkIndex(index);
		m_layerSizes[index] = size;
	}

	virtual int getLayerSize(int i) {
		return m_layerSizes[i];
	}

	virtual size_t getNumberOfLayers() {
		return m_layerSizes.size();
	}

protected:
	void checkSize(int size) {
#ifdef _NEURALNET_DEBUG
		if (size < 1) throw std::invalid_argument("size < 1 of layer\n");
#endif // _NEURALNET_DEBUG
	}
	void checkIndex(int index) {
#ifdef _NEURALNET_DEBUG
		if (index >= m_layerSizes.size() || index < 0) throw std::invalid_argument("index > topology size, use add layer\n");
#endif // _NEURALNET_DEBUG
	}
private:
	std::vector<int> m_layerSizes;
};
