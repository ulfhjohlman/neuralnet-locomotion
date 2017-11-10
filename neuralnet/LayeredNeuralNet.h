#pragma once
#include "NeuralNet.h"
#include "Topology.h"
#include "LayeredTopology.h"
#include "Layer.h"
#include "LayerFactory.h"
#include "ParameterUpdater.h"
#include "config.h"
#include <vector>
#include <stdexcept>
#include <memory>
#include <string>

//Change to "LayeredNeuralNet"
class LayeredNeuralNet :
	public NeuralNet
{
public:
	LayeredNeuralNet() : m_topology(nullptr) { }
	LayeredNeuralNet(LayeredTopology * topology) : m_topology(nullptr) {
		setTopology(topology);
		constructFromTopology();
	}
	virtual ~LayeredNeuralNet() {
		if (m_topology)
			delete m_topology;
		destroyLayers();
	}

	virtual void setTopology(LayeredTopology* topology) {
		checkTopology(topology);
		//new topology ok.
		//delete old if there is one.
		if (m_topology) {
			delete m_topology;
			std::cout << "WARNING: Replaced an old topology in LayeredNN\n";
		}
		m_topology = topology;
	}

	virtual void constructFromTopology() {
		destroyLayers();
		m_layers.reserve(m_topology->getNumberOfLayers());

		Layer* inputLayer = LayerFactory::constructLayer(m_topology->getLayerSize(0), 0, m_topology->getLayerType(0));
		m_layers.push_back(inputLayer);

		//construct network from topology
		for (int i = 1; i < m_topology->getNumberOfLayers(); i++) {
			int numberOfInputs = m_topology->getLayerSize(i - 1);
			int layerSize = m_topology->getLayerSize(i);
			int layerType = m_topology->getLayerType(i);

			Layer* newLayer = LayerFactory::constructLayer(layerSize, numberOfInputs, layerType);
			m_layers.push_back(newLayer);
		}
	}

	virtual void initializeRandomWeights() {
		for (auto & layer : m_layers)
			layer->setRandom();
	}

	virtual void cacheLayerParams()
	{
		//skips inputlayer
		for(int i=1 ; i<m_layers.size() ; i++)
		{
			m_layers[i]->cacheParamGradients();
		}
	}
	virtual void popCacheLayerParams()
	{
		//skips inputlayer
		for(int i=1 ; i<m_layers.size() ; i++)
		{
			m_layers[i]->popCacheParamGradients();
		}
	}
	virtual void reserveLayerOutputCache(int res)
	{
		try{
			for(int i = 0; i<m_layers.size(); i++) //including input_layer
			{
				m_layers[i]->reserveOutputCache(res);
				// This can endup being ALOT of space in PPO, but due to Matrixes
				// being dynamic size they arent truly allocated untill actual copy
			}
		}
		catch(std::bad_alloc e)
		{
			std::cout << "Allocation of a layers m_outputs_cache failed!\n";
			std::cout << e.what();
			std::cout << "Aborting program\n";
			exit(1);
		}
	}

	void cachePushBackOutputs()
	{
		for(int i = 0; i <m_layers.size() ;i++)
		{
			m_layers[i]->cachePushBackOutputs();
		}
	}

	void clearOutputsCache()
	{
		for(int i = 0; i <m_layers.size() ;i++)
		{
			m_layers[i]->clearOutputsCache();
		}
	}

	void uncacheIndexedOutput(int j)
	{
		for(int i = 0; i <m_layers.size() ;i++)
		{
			m_layers[i]->uncacheIndexedOutput(j);
		}
	}

	virtual void initializeXavier() {
		for (auto & layer : m_layers)
			layer->setRandomXavier();
	}

	virtual void input(const MatrixType& x) {
		checkEmptyNetwork();

		//Load input layer
		m_layers.front()->input(x);

		//Propagate forward
		for (size_t i = 1; i < m_layers.size(); i++) {
			m_layers[i]->input(m_layers[i-1]->output());
		}
	}
	virtual void backprop(const MatrixType& gradients)
	{
		checkEmptyNetwork();
		size_t second_last_index = m_layers.size()-2;
		m_layers.back()->backprop( gradients , m_layers[second_last_index]->output());
		for (size_t i = second_last_index; i > 0 ; i--)
		{
			m_layers[i]->backprop( m_layers[i+1]->getInputGradients(), m_layers[i-1]->output());
		}
	}

	//vanila SGD updates without using a gradient updator object
	// DEPRICATED; REMOVE LATER
	virtual void updateWeightsSGD(double learning_rate)
	{
		for (auto& layer : m_layers)
		{
			layer->updateWeightsSGD(learning_rate);
		}
	}

	//links a gradient updater to the network layers' parameters
	void setParameterUpdater(ParameterUpdater& gupd)
	{
		m_parameter_updater = &gupd;
		for(int i=1;i< m_layers.size();i++) //skip input layer
		{
			m_parameter_updater->linkLayerParams(m_layers[i]);
		}
	}

	void updateParameters()
	{
		m_parameter_updater->updateParameters();
	}

	virtual const MatrixType& output() {
		checkEmptyNetwork();
		return m_layers.back()->output();
	}

	virtual LayeredTopology * getTopology() const {
		return m_topology;
	}
	virtual Layer* getLayer(int i) const {
		return m_layers[i];
	}

	virtual void save(const char* path) {
		//Save this, gonna aggregate members in future here
		if (m_name == "")
			m_name = "unnamed";
		std::string data_structure_name = path + m_name;
		m_document.clear();
		m_document.insert(m_name.c_str());
		NeuralNet::save(data_structure_name.c_str());

		//save layers
		for (int i = 0; i < static_cast<int>(m_layers.size()); i++) {
			Layer* layer = m_layers[i];
			layer->setLayerIndex(i);
			layer->save(data_structure_name.c_str());
		}

		//save topology
		if (m_topology) {
			std::string topology_string = "topology";
			topology_string = data_structure_name + topology_string;
			m_topology->save(topology_string.c_str());
		} 
	}
	virtual void load(const char* path) {
		NeuralNet::load(path);
		//Load topology.
		if (!m_topology)
			m_topology = new LayeredTopology;
		std::string string_full_path = "topology";
		string_full_path = path + string_full_path;
		m_topology->load(string_full_path.c_str());

		//Allocate memory and build layer types.
		constructFromTopology();

		//Load layer data.
		for (int i = 0; i < static_cast<int>(m_layers.size()); i++) {
			Layer* layer = m_layers[i];
			layer->setLayerIndex(i);
			
			string_full_path = "layer_";
			string_full_path = path + string_full_path + std::to_string(i);

			layer->load(string_full_path.c_str());
		}
	}
protected: //members
	std::vector<Layer*> m_layers;//replace with shared_ptr<Layer>

private:   //members
	LayeredTopology* m_topology = nullptr;
public:
	ParameterUpdater* m_parameter_updater = nullptr;


	void destroyLayers()
	{
		for (size_t i = 0; i < m_layers.size(); i++)
			if (m_layers[i]) {
				//std::cout << "deleted:" << m_layers[i] << std::endl;
				delete m_layers[i];
			}
		m_layers.clear();
	}


protected: //Error checking
	void checkEmptyNetwork() const {
#ifdef _NEURALNET_DEBUG
		if (m_layers.empty()) throw NeuralNetException("Empty neural net");
#endif // _NEURALNET_DEBUG
	}
private: //Error checking
	void checkTopology(LayeredTopology* topology)
	{
#ifdef _NEURALNET_DEBUG
		if (topology == nullptr) throw std::invalid_argument("topology nullptr");
		if (topology->getNumberOfLayers() < 2) throw std::invalid_argument("input not specified.");
		if (topology == m_topology) throw std::invalid_argument("self topology assignment.");;
#endif // _NEURALNET_DEBUG
	}

public: //Print functions
	void printLayerOutputs()
	{
		std::cout << "Printing layer outputs:\n";
		for (auto& layer : m_layers)
		{
			std::cout << layer->output() << "\n\n";
		}
	}
	void printLayerWeights()
	{
		std::cout << "Printing layer weights:\n";
		for (auto& layer : m_layers)
		{
			std::cout << layer->getWeights() << "\n\n";
		}
	}
	void printLayerBias()
	{
		std::cout << "Printing layer bias:\n";
		for (auto& layer : m_layers)
		{
			std::cout << layer->getBias() << "\n\n";
		}
	}
	void printLayerWeightGradients()
	{
		std::cout << "Printing layer weight gradients:\n";
		for (auto& layer : m_layers)
		{
			std::cout << layer->getWeightGradients() << "\n\n";
		}
	}
	void printLayerInputGradients()
	{
		std::cout << "Printing layer input gradients:\n";
		for (auto& layer : m_layers)
		{
			std::cout << layer->getInputGradients() << "\n\n";
		}
	}

};

typedef LayeredNeuralNet FFNN;
