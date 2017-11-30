#pragma once

#include "Population.h"
#include "utilityfunctions.h"
#include "ThreadsafeQueue.h"

#include <algorithm>
#include <vector>
#include <numeric>

ScalarType start_niche_merge_radius = 10.0;
ScalarType start_niche_max_radius = 15.0;

ScalarType min_merge_radius = 1.0;
ScalarType min_max_radius = 2.0;

ScalarType max_niche_size = 200;
ScalarType nominal_number_of_niches = 5;

template<typename T>
class Niche 
{
public:
	Niche() = default;
	~Niche() = default;

	void addMember( std::shared_ptr< Individual<T> >& member ) {
		m_population.addMember(member);
	}
	
	ScalarType distanceTo(const Niche<T>& rhs) {
		VectorType dist = m_mid_point - rhs.m_mid_point;
		return dist.norm();
	}

	ScalarType distanceTo(const VectorType& v) {
		VectorType dist = m_mid_point - v;
		return dist.norm();
	}

	ScalarType distanceTo(std::shared_ptr< Individual<T> >& member) {
		VectorType v;
		member->getGenome()->getGeneSet(v);
		VectorType dist = m_mid_point - v;
		return dist.norm();
	}

	void merge(Niche<T>& merge_this) {
		if (merge_this.size() == 0)
			return;

		//scale up and add centers
		m_mid_point *= static_cast<ScalarType>(m_population.totalFitness());
		m_mid_point += merge_this.m_mid_point * static_cast<ScalarType>(merge_this.m_population.totalFitness());

		m_population.merge(merge_this.m_population);

		//add areas of niches
		//merge_radius = std::sqrt(merge_radius * merge_radius + merge_this.merge_radius * merge_this.merge_radius);
		//max_radius   = std::sqrt(max_radius * max_radius + merge_this.max_radius * merge_this.max_radius);

		//normalize
		m_mid_point /= static_cast<ScalarType>(m_population.totalFitness());
	}

	//moves old midpoint
	void updateMidpoint() {
		const VectorType& x_mu = m_mid_point;
		VectorType x_i, x_sum;
		x_i.resizeLike(x_mu);
		x_sum.resizeLike(x_mu);
		x_i.setZero();
		x_sum = x_i;

		for (size_t i = 0; i < m_population.size(); i++) {
			m_population[i]->getGenome()->getGeneSet(x_i);
			x_sum += (x_i - x_mu) * m_population[i]->getFitness();
		}

		m_mid_point += x_sum / m_population.totalFitness();
	}

	const void center() {
		if (m_population.size() > 0) {
			//Get first individual and weight
			m_population[0]->getGenome()->getGeneSet(m_mid_point);
			m_mid_point *= m_population[0]->getFitness();

			//add weights of others
			VectorType v;
			for (size_t i = 1; i < m_population.size(); i++) {
				m_population[i]->getGenome()->getGeneSet(v);
				m_mid_point += v * m_population[i]->getFitness();
			}

			//normalize
			m_mid_point /= m_population.totalFitness();

			//std::cout << m_mid_point << std::endl;
		}
		else {
			throw std::runtime_error("empty niche");
		}
	}

	void remove(std::shared_ptr< Individual<T> >& member) {
		m_population.remove(member);
	}

	Population<T>& getPopulation() {
		return m_population;
	}

	size_t size() {
		return m_population.size();
	}

	size_t id;
	ScalarType merge_radius = start_niche_merge_radius;
	ScalarType max_radius = start_niche_max_radius;
private:
	Population<T> m_population;
	VectorType m_mid_point;
};


template<typename T>
class NicheSet 
{
public:
	NicheSet() = default;
	~NicheSet() = default;

	void sort() {
		for (auto & niche : m_niches)
			niche.getPopulation().sort();
	}
	
	void reset(Population<T> population) {
		m_niches.clear();
		m_niches.reserve(population.size());

		
		for (size_t i = 0; i < population.size(); i++) {
			Niche<T> niche;
			niche.addMember(population[i]);
			niche.center();
			m_niches.push_back(niche);
		}
	}

	void addMember(std::shared_ptr< Individual<T> >& member) {
		VectorType v;
		member->getGenome()->getGeneSet(v);

		ScalarType min_distance = start_niche_max_radius;
		int k = 0;
		for (auto & niche : m_niches) {
			ScalarType distance = niche.distanceTo(v);

			if (distance < niche.max_radius) {
				niche.addMember(member); 
				k++; //adds to k niches
			}

			if (distance < min_distance)
				min_distance = distance;
		}

		if (k == 0) {
			Niche<T> niche;
			niche.addMember(member);
			niche.center();
			niche.merge_radius = std::min<ScalarType>( min_distance * 0.65, start_niche_merge_radius);
			niche.max_radius = std::min<ScalarType>( min_distance * 0.9, start_niche_max_radius);
			m_niches.push_back(niche);
		}
	}

	void remove(std::shared_ptr< Individual<T> >& member) {
		for (size_t i = 0; i < m_niches.size(); i++)
			m_niches[i].remove(member);
		clearEmptyNiches();
	}

	void clearEmptyNiches() {
		for (size_t i = 0; i < m_niches.size(); i++) {
			if (m_niches[i].size() == 0) {
				m_niches.erase(m_niches.begin() + i);
				i--;
			}
		}
	}

	void update() {
		ThreadsafeQueue<std::shared_ptr<Individual<T>>> move_list;
		auto dislocate = [this, &move_list](int i) {
			auto& niche = m_niches[i];
			for (int j = 0; j < niche.getPopulation().size(); j++) {
				if (niche.distanceTo(niche.getPopulation()[j]) > niche.max_radius) {
					if (niche.getPopulation()[j].use_count() == 2) {
						move_list.push(std::move(niche.getPopulation()[j]));
					} //else already in another niche
					niche.getPopulation().erase(j);
					j--;
				}
			}
		};
		parallel_for<size_t>(0, m_niches.size(), 1, dislocate);

		while (!move_list.empty()) {
			this->addMember(move_list.sequential_pop());
		}

		//Possible empty niche after last operation.
		removeEmptyNiches();

		for (auto& niche : m_niches)
			niche.updateMidpoint();

		if (m_niches.size() > nominal_number_of_niches) {
			for (auto& niche : m_niches) {
				niche.max_radius *= 1.006;
				niche.merge_radius *= 1.001;
			}
		}
		else if(m_niches.size() > nominal_number_of_niches / 2) {
			for (auto& niche : m_niches) {
				niche.max_radius *= 1.0005;
			}
		}

		//make jagged array of inter niche distances
		size_t size = m_niches.size() - 1;
		size_t it = size; //need not to know distance to self
		std::vector<std::vector <ScalarType> > niche_distances(size);
		
		for (auto & v : niche_distances) {
			v = std::vector<ScalarType>(it);
			it--;
		}

		//calculate inter niche dist
		for (size_t i = 0; i < size; i++)
			for (size_t j = i; j < size; j++) 
				niche_distances[i][j - i] = m_niches[i].distanceTo(m_niches[j+1]);
		
		ScalarType sum = 0;
		for (size_t i = 0; i < size; i++) {
			for (size_t j = i; j < size; j++) {
				//std::cout << niche_distances[i][j - i] << " ";
				sum += niche_distances[i][j - i];
			}
			//std::cout << std::endl;
		}
		std::cout << "avg niche dist=" << sum / (size * size / 2.0) << std::endl;
		//if (sum != sum) {
		//	std::cout << "FUUCK?...";
		//	std::cin.get();
		//}
		
		//sort distances BUGGG!!!!
		/*for (auto & v : niche_distances)
			std::sort(v.begin(), v.end());*/

		//merge niches, -1 last niche has already been checked by others
		for (size_t i = 0; i < size; i++) {
			for (size_t j = i; j < size; j++) {
				//std::cout << "merged" << j << " -> " << i << std::endl;
				if(niche_distances[i][j-i] < m_niches[i].merge_radius)
					m_niches[i].merge(m_niches[j+1]);
				//add check for overlapping max radius and graduall merge
			}
		}

		//Empty merged niches need to delete
		removeEmptyNiches();

		//check overfull niches
		splitHalfNiche();

		//Sort by fitness inside each niche.
		this->sort();
	}

	void splitHalfNiche()
	{
		for (size_t i = 0; i < m_niches.size(); i++)
		{
			auto& niche = m_niches[i];
			if (niche.size() > max_niche_size) {
				Niche<T> part1, part2;
				part1.merge_radius = niche.merge_radius / 2.1;
				part2.merge_radius = niche.merge_radius / 2.1;

				part1.max_radius = niche.max_radius / 2.0;
				part2.max_radius = niche.max_radius / 2.0;

				part1.merge_radius = std::max(part1.merge_radius, min_merge_radius);
				part2.merge_radius = std::max(part2.merge_radius, min_merge_radius);
				part1.max_radius = std::max(part1.max_radius, min_max_radius);
				part2.max_radius = std::max(part2.max_radius, min_max_radius);

				Population<T>& population = niche.getPopulation();
				size_t members_to_part1 = population.size() / 2;
				size_t members_to_part2 = population.size();
				for (size_t j = 0; j < members_to_part1; j++)
					part1.addMember(population[j]);
				for (size_t j = members_to_part1; j < population.size(); j++)
					part2.addMember(population[j]);
				m_niches.erase(m_niches.begin() + i);
				i--;

				part1.center();
				part2.center();

				m_niches.push_back(part1);
				m_niches.push_back(part2);

			}
		}
	}

	void removeEmptyNiches()
	{
		//Remove empty niches.
		for (size_t i = 0; i < m_niches.size(); i++) {
			//std::cout << m_niches[i].size() << std::endl;
			if (m_niches[i].size() == 0) {
				m_niches.erase(m_niches.begin() + i);
				i--;
			}
		}
	}

	void printNicheSizes() {
		for (auto & niche : m_niches)
			std::cout << niche.size() << "(" << std::setprecision(3) << niche.merge_radius << ") ";
		std::cout << std::endl;
	}

	Population<T> & operator[](size_t i) {
		return m_niches[i].getPopulation();
	}

	size_t size() {
		return m_niches.size();
	}

	void removeNiche(size_t i) {
		m_niches.erase(m_niches.begin() + i);
	}

	void sortNiches() {
		auto cmp_by_fitness = [](Niche<T>& a, Niche<T>& b) {
			return a.getPopulation()[0]->getFitness() > b.getPopulation()[0]->getFitness();
		};
		std::sort(m_niches.begin(), m_niches.end(), cmp_by_fitness);
	}
private:
	std::vector<Niche<T>> m_niches;
};

/* split code
niche.max_radius *= 0.97;
niche.merge_radius *= 0.97;

niche.merge_radius = std::max(niche.merge_radius, min_merge_radius);
niche.max_radius = std::max(niche.max_radius, min_max_radius);

*/
