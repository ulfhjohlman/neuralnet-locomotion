#pragma once
#include<memory>
#include<string>
#include "NeuralNetGenome.h"


template<typename Solution> //Add Genome template later
class Individual 
{
public:
	Individual() = default;
	virtual ~Individual() = default;
	Individual& operator = (const Individual& copy_this) {
		m_fitness = copy_this.m_fitness;
		m_id = copy_this.m_id;

		if (m_genome && copy_this.m_genome)
			*m_genome = *copy_this.m_genome;

		return *this;
	}

	virtual Solution * decode() = 0; //ha ha this works

	virtual void setFitness(double fitness) { m_fitness = fitness; }
	virtual double getFitness() const { return m_fitness; }

	virtual void save(const char* path) = 0;
	virtual void load(const char* path) = 0;

	virtual bool operator<(const Individual& rhs) const { return this->m_fitness < rhs.m_fitness; }
	virtual bool operator>(const Individual& rhs) const { return this->m_fitness > rhs.m_fitness; }

	std::shared_ptr<NeuralNetGenome> getGenome() const { return m_genome; }
protected:
	double m_fitness;
	int m_id;
	std::shared_ptr<NeuralNetGenome> m_genome;
private:
	std::string m_genome_filename;

public:
	Individual(const Individual& copy_this) = delete;

	Individual(Individual&& move_this) = delete;
	Individual& operator=(Individual&& move_this) = delete;
};
