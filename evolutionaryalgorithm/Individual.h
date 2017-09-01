#pragma once
#include<memory>
#include<string>

template<typename Genome>
class Individual 
{
public:
	Individual() = default;
	~Individual() = default;

	void decode();
	void destroyDecoding();

	void loadGenome();
	void unloadGenome();

	void save(const char* path);

	bool operator<(const Individual& rhs) const { return this->m_fitness < rhs.m_fitness; }
	bool operator>(const Individual& rhs) const { return this->m_fitness > rhs.m_fitness; }
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