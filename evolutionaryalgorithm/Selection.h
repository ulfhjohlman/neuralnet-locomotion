#pragma once
#include "Population.h"
#include "Individual.h"
#include "NeuralNetGenome.h"
#include "utilityfunctions.h"
#include "Generator.h"

#include <tuple>
#include <stdexcept>
#include <set>

class Selection {
	
};

class TournamentSelection {
public:
	TournamentSelection(ScalarType ptour, ScalarType tournament_size) : m_ptour(ptour), m_tournament_size(tournament_size){ 
		std::cout << "Tournament Selection operator:" << std::endl;
		std::cout << "ptour=" << ptour << std::endl;
		std::cout << "tournament size=" <<tournament_size << std::endl;
	}
	~TournamentSelection() = default;

	void setTournamentSelectionProbability(ScalarType ptour) {
		if (ptour < 0)
			throw std::invalid_argument("negative prob selection");

		m_ptour = ptour;
	}
	void setTournamentSize(int size) {
		if (size < 1)
			throw std::invalid_argument("too small tournament.");

		m_tournament_size = size;
	}

	template<typename T>
	std::pair<int, int> selectPair(const Population<T>& population) {
		int mate1 = this->select(population);
		int mate2 = this->select(population);
		while (mate2 == mate1) {
			mate2 = this->select(population);
		}
		return std::make_pair(mate1, mate2);
	}

	template<typename T>
	int select(const Population<T>& population) {
		std::set<int> indexes;
		Generator generator;

		//Select everyone if 
		if (population.size() < m_tournament_size)
			for (int i = 0; i < population.size(); i++)
				indexes.insert(i);

		//Select unique candidates else
		else
			while (indexes.size() < m_tournament_size) {
				int index = generator.generate_uniform_int(0, population.size() - 1);
				indexes.insert(index);
			}

		//Select best individual, else next and so on, population is assumed to be sorted
		while (indexes.size() > 1) {
			ScalarType r = generator.generate_uniform<ScalarType>(0.0f, 1.0f);
			if (r < m_ptour)
				return *indexes.begin();
			indexes.erase(indexes.begin());
		}

		return *indexes.begin();
	}


private:
	ScalarType m_ptour; 
	int m_tournament_size;
};