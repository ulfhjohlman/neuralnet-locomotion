#pragma once

#include "Layer.h"
#include <stdexcept>

class LayerFactory 
{
public:
	LayerFactory() = delete;
	~LayerFactory() = default;
	
	LayerFactory(const LayerFactory& copy_this) = delete;
	LayerFactory& operator=(const LayerFactory& copy_this) = delete;
	
	LayerFactory(LayerFactory&& move_this) = delete;
	LayerFactory& operator=(LayerFactory&& move_this) = delete;

	static Layer* constructLayer(int layerSize, int numberOfInputs, Layer::LayerType type) {
		Layer* layer = nullptr;
		switch (type)
		{
		case Layer::noActivation:
			layer = new Layer(layerSize, numberOfInputs);
			break;
		case Layer::tanh:
			break;
		case Layer::sigmoid:
			break;
		case Layer::relu:
			break;
		case Layer::inputLayer:
			break;
		case Layer::scalingLayer:
			break;
		default:
			break;
		}
		if (!layer)
			throw std::runtime_error("Unable to construct layer or allocate memory for it");

		return layer;
	}
	
private:
	
};
