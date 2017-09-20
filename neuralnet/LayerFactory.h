#pragma once

#include "Layer.h"
#include "HyperbolicTangentLayer.h"
#include "InputLayer.h"
#include "RectifiedLinearUnitLayer.h"
#include "FactoryException.h"


class LayerFactory 
{
public:
	static Layer* constructLayer(int layerSize, int numberOfInputs, int layerType) {
		Layer* layer = nullptr;

		checkLayerArgs(layerSize, numberOfInputs, layerType);
		switch (layerType)
		{
		case Layer::noActivation:
			layer = new Layer(layerSize, numberOfInputs);
			break;
		case Layer::tanh:
			layer = new HyperbolicTangentLayer(layerSize, numberOfInputs);
			break;
		case Layer::sigmoid:
			break;
		case Layer::relu:
			layer = new RectifiedLinearUnitLayer(layerSize, numberOfInputs);
			break;
		case Layer::inputLayer:
			layer = new InputLayer(layerSize, numberOfInputs); // for now
			break;
		case Layer::scalingLayer:
			break;
		case Layer::noLayer:
			break;
		default:
			break;
		}
		if (!layer)
			throw FactoryException("Unable to construct layer or allocate memory for it");

		return layer;
	}
	
private:
	static void checkLayerArgs(int layerSize, int numberOfInputs, int layerType)
	{
#ifdef _NEURALNET_DEBUG
		if (numberOfInputs < 1 && layerType != Layer::inputLayer)
			throw FactoryException("Layer with no inputs.");
		if (layerSize < 1)
			throw FactoryException("Layer with no neurons.");
		if (layerType < 0 || layerType > Layer::noLayer)
			throw FactoryException("Invalid layer type.");
#endif // _NEURALNET_DEBUG
	}

public:
	LayerFactory() = delete;
	~LayerFactory() = default;

	LayerFactory(const LayerFactory& copy_this) = delete;
	LayerFactory& operator=(const LayerFactory& copy_this) = delete;

	LayerFactory(LayerFactory&& move_this) = delete;
	LayerFactory& operator=(LayerFactory&& move_this) = delete;
};
