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

void saveTest() {
	std::vector<int> layerSizes = { 2, 128, 128, 128, 15 };
	std::vector<int> layerTypes = { Layer::inputLayer, 1, 1, 0, 0 };
	LayeredTopology* top1 = new LayeredTopology(layerSizes, layerTypes);
	LayeredNeuralNet* nn = new LayeredNeuralNet(top1);

	fs::create_directory("asd");
	nn->initializeRandomWeights();
	nn->save("asd/");
	LayeredNeuralNet * nn2 = new LayeredNeuralNet;
	nn2->load("asd/unnamed");

	Generator g;
	for (int i = 0; i < 1000; i++) {
		MatrixType m(2, 1);
		g.fill_vector_normal<float>(m.data(), m.size(), 0, 1);
		nn->input(m);
		nn2->input(m);

		float tol = 1e-10;
		if ((nn->output() - nn2->output()).norm() > tol) {
			std::cout << "failed save result test" << std::endl;
			return;
		}


	}

	std::cout << "successful save result test" << std::endl;
	std::cin.get();
	/*std::cout << nn->output() << std::endl << std::endl;
	std::cout << nn2->output() << std::endl; */
	

	delete nn;
	delete nn2;
}




int main()
{
	RandomEngineFactory::initialize(); //optional

	saveTest();

	std::vector<int> a{ 1,2,3 };
	std::vector<int> b{ 4,5,6 };

	b.insert(b.end(), 
		a.begin(), 
		a.end());

	std::sort(b.begin(), b.end());
	auto it = std::lower_bound(b.begin(), b.end(), 5);

	if(it != b.end())
		std::cout << *it << " at pos=" << it - b.begin() << std::endl;

	std::shared_ptr<int> asd1(new int), asd2(new int);
	//asd1 = asd2;
	if (asd1.get() == asd2.get())
		std::cout << "a==b \n";


	for (size_t i = 0; i < b.size(); i++)
	{
		std::cout << b[i] << std::endl;
	}

	for (size_t i = 0; i < a.size(); i++)
	{
		std::cout << a[i] << " / " << std::endl;
	}

	std::cin.get();
	return 0;
}
