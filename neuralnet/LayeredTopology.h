#pragma once
#include <vector>
#include <stdexcept>
#include <initializer_list>


class LayeredTopology 
{
public:
	LayeredTopology() = default;
	LayeredTopology(int reserveLayers) { m_layerSizes.reserve(reserveLayers); }
	LayeredTopology(std::initializer_list<int> layerSize) : m_layerSizes(layerSize) { }
	~LayeredTopology() = default;
	
	void addLayer(int size) {
		checkSize(size);
		m_layerSizes.push_back(size);
	}

	int getLayerSize(int i) {
		return m_layerSizes[i];
	}

	size_t getNumberOfLayers() {
		return m_layerSizes.size();
	}

protected:
	void checkSize(int size) {
#ifdef _DEBUG
		if (size < 1) throw std::invalid_argument("size < 1 of layer\n");
#endif // _DEBUG
	}
private:
	std::vector<int> m_layerSizes;
};
