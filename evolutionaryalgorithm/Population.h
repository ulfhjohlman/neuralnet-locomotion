#pragma once
#include <vector>
#include <iterator>
#include <memory>
#include <algorithm>

#include "Individual.h"
#include <dense>

template<typename T>
class Population 
{
public:
	typedef std::shared_ptr< Individual<T> > PopulationMember;
	//typedef std::vector<PopulationMember>::iterator PopulationIterator;

	Population() = default;
	~Population() = default;
	std::vector<PopulationMember> members;

	void addMember(PopulationMember& member) {
		members.push_back(member);
	}

	void remove( PopulationMember erase_this ) {
		size_t k = 0;
		for (auto& member : members) {
			if (member.get() == erase_this.get()) {
				members.erase(members.begin() + k);
				return;
			}
			k++;
		}
	}

	Population<T> subPopulation(size_t from, size_t to) {
		Population<T> sub_population;
		sub_population.members.insert(sub_population.members.end(), 
			members.begin() + from, 
			members.begin() + to);
		return sub_population;
	}

	void merge(Population<T>& merge_this) {
		for (auto & merge_member : merge_this.members) {
			if (std::find_if(members.begin(), members.end(),
				[&merge_member](const PopulationMember& this_member)
			{ return merge_member.get() == this_member.get(); })  == members.end()) {
				this->members.insert(members.end(), std::move(merge_member));
			}
		}
		merge_this.members.clear();
	}

	//descending sort
	void sort() {
		auto cmp_by_fitness = [](const std::shared_ptr<Individual<T>>& a, const std::shared_ptr<Individual<T>>& b) {
			return a->getFitness() > b->getFitness();
		};
		std::sort(members.begin(), members.end(), cmp_by_fitness);
	}
	void save(int generation, const char* name, int n_best = 1) {
		std::string s = "generation" + std::to_string(generation) + "/";
		std::experimental::filesystem::create_directory(s.c_str());

		for (int i = 0; i < n_best; i++) {
			members[i]->decode()->setName(name);
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

	double totalFitness() const {
		double sum = 0;
		for (const auto & member : members)
			sum += member->getFitness();
		return sum;
	}

	void erase(int index) {
		members.erase(members.begin() + index);
	}

	PopulationMember back() {
		return members.back();
	}
	
	size_t size() const {
		return members.size();
	}

	void clearMutationFlag() {
		for (auto & i : members)
			i->getGenome()->clearMutationFlag();
	}

protected:

private:
	//Population(const Population& copy_this) = default;
	//Population& operator=(const Population& copy_this) = delete;
};
