#pragma once
#include "LayeredTopology.h"


class CascadeTopology : public LayeredTopology
{
public:
	CascadeTopology() = default;
	CascadeTopology(const std::vector<int>& layerSize,
					const std::vector<int>& layerType)
					: LayeredTopology(layerSize, layerType) {  }
	CascadeTopology(std::initializer_list<int> layerSize,
					std::initializer_list<int> layerType)
					: LayeredTopology(layerSize, layerType) { }
	virtual ~CascadeTopology() = default;

	virtual void addLayerConnection(int index,
		std::initializer_list<int> layerSize) {

		std::vector<int> tmp(layerSize);
		addLayerConnection(index, tmp);
	}

	virtual void addLayerConnection(int index, 
		const std::vector<int>& layerConnection) {
		checkConnection(index, layerConnection);

		if (m_layerConnections.size() < 1)
			addDummyNode();
		m_layerConnections.push_back(layerConnection);
	}
	virtual void removeLayer(int index) {
		checkIndex(index);
		LayeredTopology::removeLayer(index);
		m_layerConnections.erase(m_layerConnections.begin() + index);
	}

	const std::vector<int>& getLayerConnections(int i) {
		return m_layerConnections[i];
	}
protected:
	inline void checkConnection(int index, const std::vector<int>& layerConnection) {
#ifdef _NEURALNET_DEBUG
		bool indexOk = index < this->getNumberOfLayers();
		if (!indexOk)
			throw std::invalid_argument("layer index out of bounds.");

		auto validConnection = [](const std::vector<int>& vec, int index, int maxLayer)
		{ 
			return std::all_of(vec.begin(), vec.end(), [maxLayer, index](int i) { 
				return i >-1 && i < index; 
			});
		};

		size_t nLayers = this->getNumberOfLayers();
		bool connectionOk = validConnection(layerConnection, index, nLayers);

		if (!connectionOk)
			throw std::invalid_argument("Connection not valid.");
#endif // _NEURALNET_DEBUG
	}
	
private:

	void addDummyNode() {
		m_layerConnections.push_back({ 0 });
	}

	std::vector<std::vector<int>> m_layerConnections;
};