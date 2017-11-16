#pragma once

#include "Population.h"

#include <algorithm>
#include <vector>
#include <numeric>

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
		auto dist = m_mid_point - rhs.m_mid_point;
		return dist.norm();
	}

	ScalarType distanceTo(const VectorType& v) {
		auto dist = m_mid_point - v;
		return dist.norm();
	}

	void merge(Niche<T>& merge_this) {
		//scale up and add centers
		m_mid_point *= static_cast<ScalarType>(m_population.totalFitness());
		m_mid_point += merge_this.m_mid_point * static_cast<ScalarType>(merge_this.m_population.totalFitness());

		m_population.merge(merge_this.m_population);

		//normalize
		m_mid_point /= static_cast<ScalarType>(m_population.totalFitness());
	}

	//moves old midpoint
	void updateMidpoint() {
		const VectorType& x_mu = m_mid_point;
		VectorType x_i, x_sum;
		x_i.resizeLike(x_mu);
		x_i.setZero();
		x_sum = x_i;

		for (size_t i = 0; i < m_population.size(); i++) {
			m_population[i]->getGenome()->getGeneSet(x_i);
			x_sum += (x_i - x_mu) * m_population[i]->getFitness();
		}

		m_mid_point += x_sum / m_population.totalFitness();
	}

	const VectorType& center() {
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

		int k = 0;
		for (auto & niche : m_niches) {
			if (niche.distanceTo(v) < max_niche_radius) {
				niche.addMember(member);
				k++; //adds to k niches
			}
		}

		if (k == 0) {
			Niche<T> niche;
			niche.addMember(member);
			niche.center();
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
		for (auto& niche : m_niches)
			niche.updateMidpoint();

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
		std::cout << "avg niche dist=" << sum / (size * size / 2) << std::endl;
		
		//sort distances
		for (auto & v : niche_distances)
			std::sort(v.begin(), v.end());

		//merge niches, -1 last niche has already been checked by others
		for (size_t i = 0; i < m_niches.size()-1; i++) {
			size_t pos = 0;
			if (niche_distances[i].back() < min_niche_radius)
				pos = std::distance(niche_distances[i].begin(), niche_distances[i].end());
			else {
				auto it = std::lower_bound(
					niche_distances[i].begin(),
					niche_distances[i].end(),
					min_niche_radius);
				
				if (it != niche_distances[i].end())
					pos = std::distance(niche_distances[i].begin(), it);
			}

			for (size_t j = i + 1; j < i + pos + 1; j++) {
				std::cout << "merged" << j << " -> " << i << std::endl;
				m_niches[i].merge(m_niches[j]);
			}

			i += pos; //might be 1 to little
		}

		//Remove empty niches.
		for (size_t i = 0; i < m_niches.size(); i++) {
			//std::cout << m_niches[i].size() << std::endl;
			if (m_niches[i].size() == 0) {
				m_niches.erase(m_niches.begin() + i);
				i--;
			}
		}

		//check overfull niches
		//not implemented yet
	}

	void printNicheSizes() {
		for (auto & niche : m_niches)
			std::cout << niche.size() << " ";
		std::cout << std::endl;
	}

	Population<T> & operator[](size_t i) {
		return m_niches[i].getPopulation();
	}

	size_t size() {
		return m_niches.size();
	}

	ScalarType min_niche_radius = 10;
	ScalarType max_niche_radius = 25;
private:
	std::vector<Niche<T>> m_niches;
};
