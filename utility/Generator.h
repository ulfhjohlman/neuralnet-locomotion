#pragma once
#include "RandomEngineFactory.h"
#include <thread>
#include <memory>
#include <vector>

//Don't abuse static functions as locking cause blocking. Make instance instead if
//much generation is needed.
class Generator 
{
public:
	Generator() {
		m_engine = RandomEngineFactory::requestEngine();
	}
	~Generator() {
		RandomEngineFactory::returnEngine(std::move(m_engine));
	}

	//1 uniform in [a,b]
	template<typename T>
	T generate_uniform(const T a, const T b) {
		std::uniform_real_distribution<T> distribution(a, b);
		return distribution(*m_engine);
	}

	//1 normal with mean = mu, sd = sigma (1 sd = sqrt VAR[X])
	template<typename T>
	T generate_normal(const T mu, const T sigma) {
		std::normal_distribution<T> distribution(mu, sigma);
		return distribution(*m_engine);
	}

	//Uniform vector
	template<typename T>
	void fill_vector_uniform(std::vector<T>& v, const T a, const T b) {
		fill_vector_uniform<T>(v.data(), v.size(), a, b);
	}
	template<typename T>
	void fill_vector_uniform(T* v, const int size, const T a, const T b) {
		std::uniform_real_distribution<T> distribution(a, b); //remove
		for (int i = 0; i < size; i++)
			v[i] = distribution(*m_engine); //Change to =generate_uniform<T>(a, b)
	}

	//Normal dist vector
	template<typename T>
	void fill_vector_normal(std::vector<T>& v, const T mu, const T sigma) {
		fill_vector_normal<T>(v.data(), v.size(), mu, sigma);
	}
	template<typename T>
	void fill_vector_normal(T* v, const int size, const T mu, const T sigma) {
		std::normal_distribution<T> distribution(mu, sigma); //remove
		for (int i = 0; i < size; i++)
			v[i] = distribution(*m_engine); //Change to =generate_normal<T>(mu, sigma)
	}

	//generate X~B(n, p), p can only be double.
	int generate_binomial(const int n, const double p) {
		std::binomial_distribution<int> distribution(n, p);
		return distribution(*m_engine);
	}
	//generate int X ~ { a, a+1, ..., b } 
	int generate_uniform_int(const int a, const int b) {
		std::uniform_int_distribution<int> distribution(a, b);
		return distribution(*m_engine);
	}

	//Add template to fix these pesky warnings.
	void fill_vector_uniform_int(std::vector<int>& v, const int a, const int b) {
		fill_vector_uniform_int(v.data(), v.size(), a, b);
	}

	void fill_vector_uniform_int(int* v, const int size, const int a, const int b) {
		for (int i = 0; i < size; i++)
			v[i] = generate_uniform_int(a, b);
	}


private:
	std::unique_ptr<RandomEngine> m_engine;

public: //Global class methods.
	template<typename T>
	static T generate_uniform_shared(const T a, const T b) {
		std::lock_guard<std::mutex> lk(c_mutex_g);
		std::uniform_real_distribution<T> distribution(a, b);
		return distribution(c_engine_g);
	}

	template<typename T>
	static T generate_normal_shared(const T mu, const T sigma) {
		std::lock_guard<std::mutex> lk(c_mutex_g);
		std::normal_distribution<T> distribution(mu, sigma);
		return distribution(c_engine_g);
	}
	
private: //global class variables
	static RandomEngine c_engine_g;
	static std::mutex c_mutex_g;
};

//Init static class variables
//Make a copy and drop unique ptr.
RandomEngine Generator::c_engine_g = *RandomEngineFactory::requestEngine();
std::mutex Generator::c_mutex_g;

namespace example {
	void generator_example() {
		Generator g;
		float f = g.generate_normal(5.0f, 2.0f); //ok
		f = g.generate_normal<float>(5.0f, 2.0f); //explicit template, ok

		//double d = g.generate_uniform(-1, 1); //not ok, cannot instantiate int_dist of real numbers and cast to double
		double d = g.generate_uniform<double>(-1, 1); //ok, implicit conversion to double of args

		std::cout << "f~N(5,2)=" << f << std::endl;
		std::cout << "d~U(-1,1)=" << d << std::endl;

		const int N = 10;
		float pf[N];
		g.fill_vector_normal(pf, N, 5.0f, 1.0f); //ok
		std::cout << N << " ~N(5,2): ";
		for (int i = 0; i < N; i++) {
			std::cout << pf[i] << " ";
		}
		std::cout << std::endl;

		auto global_d = Generator::generate_normal_shared<double>(5.0f, 1); //double
		std::cout << "Thread safe shared generation: " << global_d << std::endl;
 	}
}