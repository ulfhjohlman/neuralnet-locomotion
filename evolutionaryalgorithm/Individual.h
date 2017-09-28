#pragma once
#include<memory>
#include
#include<string>


class Individual 
{
public:
	Individual() = default;
	virtual ~Individual() = default;

	virtual void decode() = 0;
	virtual void destroyDecoding() = 0;

	virtual void loadGenome() = 0;
	virtual void unloadGenome() = 0;

	virtual void save(const char* path);

	virtual bool operator<(const Individual& rhs) const { return this->m_fitness < rhs.m_fitness; }
	virtual bool operator>(const Individual& rhs) const { return this->m_fitness > rhs.m_fitness; }
protected:
	double m_fitness;
	int m_rank;
	int m_id;
	std::shared_ptr<Genome> m_genome;
private:
	std::string m_genome_filename;

public:
	Individual(const Individual& copy_this) = delete;
	Individual& operator=(const Individual& copy_this) = delete;

	Individual(Individual&& move_this) = delete;
	Individual& operator=(Individual&& move_this) = delete;
};
