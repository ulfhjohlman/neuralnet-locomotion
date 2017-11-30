#pragma once
#include "NeuralNet.h"
#include "LayeredNeuralNet.h"
#include "Genome.h"
#include "Generator.h"
#include "dense"

#include <stdexcept>
#include <unordered_set>

const ScalarType min_variance = 1e-6f;
const ScalarType max_variance = 2.0f;

const ScalarType min_moment = -0.5f;
const ScalarType max_moment = 0.5f;

const ScalarType variance_gain = 0.004f; //0.005
const ScalarType variance_mutation = 0.02f; //0.02

const ScalarType start_variance_variance = 0.25f; //0.5
const ScalarType start_variance_mean = 0.35; //0.5

const ScalarType decay_factor = 0.9f; //momentum decay
const ScalarType eta = 0.33f; //momentum gain.



class NeuralNetGenome : public Genome
{
public:
	NeuralNetGenome(LayeredNeuralNet * net) {
		checkNN(net);

		LayeredTopology* top = net->getTopology();
		size_t number_of_layers = top->getNumberOfLayers();

		this->reserve(number_of_layers);

		//Hook up data pointers for modification
		for (size_t i = 1; i < number_of_layers; i++) {
			//Get neuron layer size
			m_layer_size.push_back(top->getLayerSize(i));
			
			//Get memory location of data.
			Layer* layer = net->getLayer(i);
			m_weight_data.push_back(&layer->getWeights());
			m_bias_data.push_back(&layer->getBias());
		}

		Generator generator;
		for (size_t i = 0; i < m_layer_size.size(); i++) {
			const int n_weight = m_weight_data[i]->size();
			const int n_bias = m_bias_data[i]->size();
			VectorType vWeight(n_weight);
			VectorType vBias(n_bias);
			vWeight.setZero();
			vBias.setZero();

			m_moment_weight.push_back(vWeight);
			m_moment_bias.push_back(vBias);
			
			generator.fill_vector_normal<ScalarType>(vWeight.data(), vWeight.size(), start_variance_mean, start_variance_variance);
			generator.fill_vector_normal<ScalarType>(vBias.data(), vBias.size(), start_variance_mean, start_variance_variance);
			
			m_mutation_variance_weights.push_back(vWeight);
			m_mutation_variance_bias.push_back(vBias);
		}

	}

	virtual ~NeuralNetGenome() = default;

	NeuralNetGenome& operator=(const NeuralNetGenome& copy_this) {
		//std::cout << "Neural genome copy" << std::endl;
		for (int i = 0; i < m_weight_data.size(); i++) {
			*m_weight_data[i] = *copy_this.m_weight_data[i];
			*m_bias_data[i] = *copy_this.m_bias_data[i];
			// m_layer_size; // identical

			m_moment_weight[i] = copy_this.m_moment_weight[i];
			m_moment_bias[i] = copy_this.m_moment_bias[i];
			m_mutation_variance_weights[i] = copy_this.m_mutation_variance_weights[i];
			m_mutation_variance_bias[i] = copy_this.m_mutation_variance_bias[i];
		}
		return *this;
	}

	virtual void uniformCrossover(const NeuralNetGenome& mate) {
		Generator generator;
		for (int i = 0; i < m_weight_data.size(); i++) {
			mixGenes(generator, *m_weight_data[i], *mate.m_weight_data[i]);
			mixGenes(generator, *m_bias_data[i], *mate.m_bias_data[i]);

			mixGenes(generator, m_moment_weight[i], mate.m_moment_weight[i]);
			mixGenes(generator, m_moment_bias[i], mate.m_moment_bias[i]);

			mixGenes(generator, m_mutation_variance_weights[i], mate.m_mutation_variance_weights[i]);
			mixGenes(generator, m_mutation_variance_bias[i], mate.m_mutation_variance_bias[i]);
		}
	}

	virtual void directionalCrossover(const NeuralNetGenome& mate) {
		ScalarType r =  Generator::generate_uniform_shared<ScalarType>(0, 1);
		for (int i = 0; i < m_weight_data.size(); i++) {
			*m_weight_data[i] += r*(*mate.m_weight_data[i] - *m_weight_data[i]);
			*m_bias_data[i]   += r*(*mate.m_bias_data[i] - *m_bias_data[i]);

			m_moment_weight[i] += r*(mate.m_moment_weight[i] - m_moment_weight[i]);
			m_moment_bias[i]   += r*(mate.m_moment_bias[i] - m_moment_bias[i]);

			m_mutation_variance_weights[i] += r*(mate.m_mutation_variance_weights[i] - m_mutation_variance_weights[i]);
			m_mutation_variance_bias[i]    += r*(mate.m_mutation_variance_bias[i] - m_mutation_variance_bias[i]);
		}
	}

	//bottleneck in large networks
	void mixGenes(Generator &generator, MatrixType& genes, const MatrixType &mateGenes)
	{
		MatrixType rolls;
		rolls.resizeLike(genes);
		generator.fill_vector_uniform<ScalarType>(rolls.data(), rolls.size(), 0.0f, 1.0f);
		genes = (rolls.array() < static_cast<ScalarType>(0.5)).select(genes, mateGenes);
	}

	//bottleneck in large networks
	void mixGenes(Generator &generator, VectorType& genes, const VectorType &mateGenes)
	{
		VectorType rolls;
		rolls.resizeLike(genes);
		generator.fill_vector_uniform<ScalarType>(rolls.data(), rolls.size(), 0.0f, 1.0f);
		genes = (rolls.array() < static_cast<ScalarType>(0.5)).select(genes, mateGenes);
	}

	virtual void decayMomentum() {
		if ( !m_atomic_exclusion.exchange(false))
			return; //already mutated.

		for (int i = 0; i < m_moment_weight.size(); i++) {
			m_moment_weight[i].array() *= static_cast<ScalarType>(.95);
			m_moment_bias[i].array() *= static_cast<ScalarType>(.95);

			m_mutation_variance_weights[i].array() *= static_cast<ScalarType>(.99);
			m_mutation_variance_weights[i].array().max(static_cast<ScalarType>(min_variance));

			m_mutation_variance_bias[i].array() *= static_cast<ScalarType>(.99);
			m_mutation_variance_bias[i].array().max(static_cast<ScalarType>(min_variance));
		}
	}

	virtual void mutate(ScalarType mutation_probability) {
		if ( !m_atomic_exclusion.exchange(false))
			return; //already mutated.

		checkMutationRate(mutation_probability);

		Generator generator;
		for (int i = 0; i < m_layer_size.size(); i++) {
			const int weight_size = m_weight_data[i]->size();
			const int bias_size = m_bias_data[i]->size();

			auto weightPointer = m_weight_data[i];
			auto biasPointer = m_bias_data[i];

			auto weightMomentum = m_moment_weight[i].data();
			auto biasMomentum = m_moment_bias[i].data();

			auto varianceWeights = m_mutation_variance_weights[i].data();
			auto varianceBias = m_mutation_variance_bias[i].data();

			/*std::cout << m_mutation_resistance[i].mean() << " ";
			std::cout << m_moment_bias[i].mean() << " ";
			std::cout << m_moment_weight[i].mean() << std::endl;*/

			//Get the binomially distributed variable ~B(n_trails, p_probability)
			int number_of_mutations_weights = generator.generate_binomial(weight_size, (double)mutation_probability);
			int number_of_mutations_bias = generator.generate_binomial(bias_size, (double)mutation_probability);

			creepMutation(number_of_mutations_weights, generator, weight_size, weightPointer->data(), weightMomentum, varianceWeights);
			creepMutation(number_of_mutations_bias, generator, bias_size, biasPointer->data(), biasMomentum, varianceBias);
		}
	}

	void creepMutation(int number_of_mutations, Generator &generator, const  int size, ScalarType * data, ScalarType * momentum, ScalarType * variance)
	{
		//Generate mutation sites. Assumed low mutation prob for now.
		//Problem is worst case scenario when same index is rolled.
		//Roll from shrinking list for high p_mut
		std::unordered_set<int> indexes(number_of_mutations);
		while (indexes.size() < number_of_mutations) {
			int index = generator.generate_uniform_int(0, size - 1);
			indexes.insert(index);
		}

		//for each mutation site
		for (const auto & index : indexes) {
			//update mutate variance
			variance[index] += generator.generate_normal<ScalarType>(variance_gain, variance_mutation);
			variance[index] = std::max<ScalarType>(min_variance, variance[index]);
			variance[index] = std::min<ScalarType>(max_variance, variance[index]);

			//weight mutation
			auto mutation_distance = generator.generate_normal<ScalarType>(momentum[index], variance[index]); //0.025 default
			data[index] += mutation_distance;

			momentum[index] += eta*mutation_distance;
			momentum[index] *= decay_factor;
			momentum[index] = std::max<ScalarType>(min_moment, momentum[index]);
			momentum[index] = std::min<ScalarType>(max_moment, momentum[index]);
		}
	}

	virtual int getNumberOfInputs() {
		return m_layer_size.front();
	}
	virtual int getNumberOfOutputs() {
		return m_layer_size.back();
	}

	void getGeneSet(VectorType& v) const {
		Eigen::Index weight_genome_size = 0;
		Eigen::Index bias_genome_size = 0;
		for (int i = 0; i < m_weight_data.size(); i++) {
			weight_genome_size += m_weight_data[i]->size();
			bias_genome_size += m_bias_data[i]->size();
		}
		Eigen::Index genome_size = weight_genome_size + bias_genome_size;
		if(v.size() != genome_size)
			v.resize(genome_size);

		Eigen::Index it = 0;
		for (int i = 0; i < m_weight_data.size(); i++) {
			Eigen::Index n = m_weight_data[i]->size(); //n genes in layer i
			v.segment(it, n) = Eigen::Map<VectorType>((*m_weight_data[i]).data(), n);
			it += n;

			n = m_bias_data[i]->size();
			v.segment(it, n) = Eigen::Map<VectorType>((*m_bias_data[i]).data(), n);
			it += n;
		}
		/*std::cout << v.transpose() << std::endl;
		std::cin.get();*/
	}

	void clearMutationFlag() {
		m_atomic_exclusion.store(true);
	}

protected:
	void reserve(size_t number_of_layers)
	{
		m_bias_data.reserve(number_of_layers);
		m_weight_data.reserve(number_of_layers);
		m_layer_size.reserve(number_of_layers);

		m_moment_bias.reserve(number_of_layers);
		m_moment_weight.reserve(number_of_layers);

		m_mutation_variance_weights.reserve(number_of_layers);
		m_mutation_variance_bias.reserve(number_of_layers);
	}


	void checkMutationRate(ScalarType mutation_probability) {
#ifdef _NEURALNET_DEBUG
		if (mutation_probability < 0.0 || mutation_probability > 0.7) {
			throw std::invalid_argument("Mutation probability not in [0, 0.7]");
		}
#endif // _DEBUG
	}
	void checkNN(LayeredNeuralNet * net)
	{
#ifdef _NEURALNET_DEBUG
		if (net == nullptr)
			throw std::runtime_error("NeuralNetGenome received nullptr NN.");
#endif // _NEURALNET_DEBUG
	}

private:
	std::vector<MatrixType*> m_weight_data; //dangling pointer danger
	std::vector<MatrixType*> m_bias_data; //dangling pointer danger

	std::vector<VectorType> m_moment_weight;
	std::vector<VectorType> m_moment_bias;

	std::vector<VectorType> m_mutation_variance_weights;
	std::vector<VectorType> m_mutation_variance_bias;

	std::vector<int>	     m_layer_size;

	std::atomic_bool m_atomic_exclusion = true;
};