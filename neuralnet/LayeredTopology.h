#pragma once
#include "Topology.h"
#include "Layer.h"

#include <vector>
#include <stdexcept>
#include <initializer_list>

class LayeredTopology : private Topology
{
public:
	LayeredTopology() = default;
	LayeredTopology(const std::vector<int>& layerSize,
					const std::vector<int>& layerType )
					: m_layerSizes(layerSize), m_layerTypes(layerType) {
		checkMatchingLayerSize(m_layerSizes, m_layerTypes);
	}

	LayeredTopology(std::initializer_list<int> layerSize,
					std::initializer_list<int> layerType )
					: LayeredTopology(std::vector<int>(layerSize),
									  std::vector<int>(layerType)) { }
	virtual ~LayeredTopology() = default;

	virtual void reserveLayers(int numberOfLayers) {
		m_layerSizes.reserve(numberOfLayers);
		m_layerTypes.reserve(numberOfLayers);
	}
	virtual void addLayer(int size, Layer::LayerType layerType = Layer::noActivation) {
		checkSize(size);

		m_layerSizes.push_back(size);
		m_layerTypes.push_back(layerType);
	}
	virtual void insert(int index, int size, Layer::LayerType layerType = Layer::noActivation) {
		checkSize(size);
		checkIndex(index);

		m_layerSizes.emplace(m_layerSizes.begin(), size);
		m_layerTypes.emplace(m_layerTypes.begin(), layerType);
	}
	virtual void setLayerType(int index, Layer::LayerType layerType) {
		checkIndex(index);

		m_layerTypes[index] = layerType;
	}

	virtual void removeLayer(int index) {
		checkIndex(index);

		m_layerSizes.erase(m_layerSizes.begin()+index);
		m_layerTypes.erase(m_layerTypes.begin()+index);
	}
	virtual void resizeLayer(int index, int size) {
		checkSize(size);
		checkIndex(index);

		m_layerSizes[index] = size;
	}

	virtual int getLayerSize(int i) {
		return m_layerSizes[i];
	}
	virtual int getLayerType(int i) {
		return m_layerTypes[i];
	}

	virtual size_t getNumberOfLayers() {
		return m_layerSizes.size();
	}
	virtual bool equals(const LayeredTopology& othertop)
	{
		for(int i =0;i<m_layerSizes.size();i++)
		{
			if(m_layerSizes[i] != othertop.m_layerSizes[i])
			{
				return false;
			}
			if(m_layerTypes[i] != othertop.m_layerTypes[i])
			{
				return false;
			}
		}
		return true;
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
	void checkMatchingLayerSize(const std::vector<int>& v1,
		const std::vector<int>& v2) {
#ifdef _NEURALNET_DEBUG
		if (v1.size() != v2.size())
			throw std::invalid_argument("layer vectors mismatch.");
#endif // _NEURALNET_DEBUG
	}
private:
	std::vector<int> m_layerSizes;
	std::vector<int> m_layerTypes;
};
