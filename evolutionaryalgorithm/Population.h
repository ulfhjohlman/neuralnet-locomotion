#pragma once
#include <vector>
#include <iterator>
#include <memory>
#include <algorithm>

#include "Individual.h"

template<typename T>
class Population 
{
public:
	typedef std::unique_ptr< Individual<T> > PopulationMember;
	//typedef std::vector<PopulationMember>::iterator PopulationIterator;

	Population() = default;
	~Population() = default;
	std::vector<PopulationMember> members;

	Population subPopulation(size_t from, size_t to) {
		Population<T> sub_population;
		sub_population.members.insert(sub_population.members.end(), 
			std::make_move_iterator(members.begin() + from), 
			std::make_move_iterator(members.begin() + to));
		return sub_population;
	}

	/*PopulationIterator begin() {
		return members.begin();
	}
	PopulationIterator end() {
		return members.end();
	}*/

	//descending sort
	void sort() {
		auto cmp_by_fitness = [](const std::unique_ptr<Individual<LayeredNeuralNet>>& a, const std::unique_ptr<Individual<LayeredNeuralNet>>& b)
		{
			return a->getFitness() > b->getFitness();
		};
		std::sort(members.begin(), members.end(), cmp_by_fitness);
	}
	void save(int generation, const char* name, int n_best = 1) {
		std::string s = "generation" + std::to_string(generation) + "/";
		std::experimental::filesystem::create_directory(s.c_str());
		s += name;
		for (int i = 0; i < n_best; i++)
		{
			s += "_" + std::to_string(i);
			members[i]->decode()->save(s.c_str());
		}
		
	}

	PopulationMember& operator[](size_t i) {
		return members[i];
	}

	double meanFitness() const {
		double mean = 0;
		for (const auto & member : members)
			mean += member->getFitness();
		mean /= static_cast<double>(members.size());
		return mean;
	}

	void erase(int index) {
		members.erase(members.begin() + index);
	}
	
	size_t size() const {
		return members.size();
	}

protected:

private:
	Population(const Population& copy_this) = delete;
	Population& operator=(const Population& copy_this) = delete;
};
