#include <iostream>

#include "RandomEngineFactory.h"
#include "Generator.h"
#include "utilityfunctions.h"
#include "ThreadsafeQueue.h"

#include "NeuralNetGenome.h"
#include "Layer.h"
#include "NeuralNetChromosome.h"
#include "Mutation.h"
#include "Population.h"

#include "tinyxml2.h"
#include "XMLFile.h"
#include "BinaryFile.h"
#include "LayeredTopology.h"

#include <future>
#include <thread>

#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;


int main()
{
	RandomEngineFactory::initialize(); //optional

	std::vector<int> layerSizes = { 2, 128, 128, 128, 15 };
	std::vector<int> layerTypes = { Layer::inputLayer, 1, 1, 0, 0 };
	LayeredTopology* top1 = new LayeredTopology(layerSizes, layerTypes);
	LayeredNeuralNet* nn = new LayeredNeuralNet(top1);

	fs::create_directory("asd");
	nn->initializeRandomWeights();
	nn->save("asd/");

	MatrixType m(2, 1);
	m(0, 0) = 1;
	m(1, 0) = 1;
	nn->input(m);

	LayeredNeuralNet nn2;
	nn2.load("asd/unnamed");
	nn2.input(m);

	std::cout << nn->output() << std::endl << std::endl;
	std::cout << nn2.output() << std::endl;

	std::cin.get();
	return 0;
}
