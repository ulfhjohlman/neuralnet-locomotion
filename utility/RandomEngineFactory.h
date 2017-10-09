#pragma once
#include <thread>
#include <mutex>
#include <queue>
#include <vector>
#include <random>
#include <numeric>
#include <memory>
#include <chrono>

#include <iostream>
#include <string>

//typedef std::ranlux48 RandomEngine;			//slower, 48-bit advancement, good quality, has theory
typedef std::ranlux24 RandomEngine;		//slower, 24-bit advancement, good quality, has theory
//typedef std::mt19937_64 RandomEngine;		//slow, big on memory, 64-bit, medium quality
//typedef std::mt19937 RandomEngine;		//slow, big on memory, 32-bit, medium quality
//typedef std::minstd_rand RandomEngine;	//fast, but poor quality

//This way ensures mainstream routines for large scale random number generation.
//Is thread safe. Singleton. Break it by returning null engine ptr, don't do that.
//No need to return engine. But it will likely help performance.
class RandomEngineFactory 
{
public:
	RandomEngineFactory() = delete;
	~RandomEngineFactory() = default;

	//You better return it !
	static std::unique_ptr< RandomEngine > requestEngine() {
		std::lock_guard<std::mutex> lk(c_mutex_rf);

		if (c_engines_rf.empty())
			constructNewEngine();

		auto holder = std::move(c_engines_rf.front());
		c_engines_rf.pop();
		return holder;
	}

	//Thank you, come again.
	static void returnEngine(std::unique_ptr< RandomEngine > pEngine) {
		std::lock_guard<std::mutex> lk(c_mutex_rf);
		c_engines_rf.push(std::move(pEngine));
	}

	//Not needed, but can be used to make good seeds and build generators.
	static void initialize() {
		std::lock_guard<std::mutex> lk(c_mutex_rf);

		//Take some random sources and feed them to seed_seq
		unsigned int seed1 = static_cast<unsigned int> (std::chrono::system_clock::now().time_since_epoch().count());
		unsigned int seed2 = c_device_rf();
		//Make a ping packet response time, thread wait time, process id
		std::seed_seq seq = { seed1, seed2 }; // consume entropy

		//This line might need tweaking on cluster. Do process get access to N cores or?
		unsigned int number_of_threads = std::thread::hardware_concurrency();
		std::vector<unsigned int> seeds(number_of_threads);
		//Generate seeds from randomness sources.
		seq.generate(seeds.begin(), seeds.end());

		//Construct a engine for each available core on system.
		for (const auto & seed : seeds) {
			std::unique_ptr<RandomEngine> ptr(new RandomEngine(seed));
			c_engines_rf.push(std::move(ptr));
		}
	}
private:

	//Not thread safe without lock m_mutex first
	static void constructNewEngine() {
		unsigned int seed = c_device_rf(); //This one is magic
		std::unique_ptr<RandomEngine> ptr(new RandomEngine(seed));
		c_engines_rf.push(std::move(ptr));
	}
	
private:
	static std::random_device							c_device_rf;
	static std::mutex									c_mutex_rf;
	static std::condition_variable						c_cond_rf;
	static std::queue<std::unique_ptr<RandomEngine>>	c_engines_rf;
};

//static init
std::mutex									RandomEngineFactory::c_mutex_rf;
std::condition_variable						RandomEngineFactory::c_cond_rf;
std::queue<std::unique_ptr<RandomEngine>>	RandomEngineFactory::c_engines_rf;
std::random_device							RandomEngineFactory::c_device_rf;

namespace example {

	//basic function template as example, can be used to fill arg vector.
	//example::generateCanonicalNumbers<float>(std::vector<float>(5));
	template<typename T>
	void generateCanonicalNumbers(std::vector<T>& v) {
		auto pEngine = RandomEngineFactory::requestEngine();

		//lambda : [pass local variables here (&ref)/(=value) ] (args) { body };
		auto generator = [&pEngine/*by reference*/] () {
			return std::generate_canonical<T, std::numeric_limits<T>::digits>(*pEngine);
		}; //Purpose is to bind arguments to generator function.

		std::generate(v.begin(), v.end(), generator);

		std::cout << "generated vector: \n";
		for (const auto& i : v)
			std::cout << i << std::endl;

		RandomEngineFactory::returnEngine(std::move(pEngine));
	}

	void uniformrealdist_hist_example() {
		const int nrolls = 1000;  // number of experiments
		const int nstars = 95;     // maximum number of stars to distribute
		const int nintervals = 10; // number of intervals

		auto generator_ptr = RandomEngineFactory::requestEngine();
		std::uniform_real_distribution<double> distribution(0.0, 1.0);

		int p[nintervals] = {};

		for (int i = 0; i < nrolls; ++i) {
			double number = distribution(*generator_ptr);
			++p[int(nintervals*number)];
		}

		std::cout << "uniform_real_distribution (0.0,1.0):" << std::endl;
		std::cout << std::fixed; std::cout.precision(1);

		for (int i = 0; i < nintervals; ++i) {
			std::cout << float(i) / nintervals << "-" << float(i + 1) / nintervals << ": ";
			std::cout << std::string(p[i] * nstars / nrolls, '*') << std::endl;
		}
		RandomEngineFactory::returnEngine(std::move(generator_ptr));
	}

	//make a normal dist and plot histogram.
	void normaldist_histogram_example() {
		const int nrolls = 2000;  // number of experiments
		const int nstars = 200;    // maximum number of stars to distribute
		const int bins = 12;

		auto generator_ptr = RandomEngineFactory::requestEngine();
		std::normal_distribution<double> distribution(6.0, 2.5);

		int p[bins] = {};

		for (int i = 0; i < nrolls; ++i) {
			double number = distribution(*generator_ptr); //magic happens on this line
			if ((number >= 0.0) && (number < bins)) ++p[int(number)];
		}

		std::cout << "normal_distribution (6.0,2.5):" << std::endl;

		for (int i = 0; i < bins; ++i) {
			std::cout << i << "-" << (i + 1) << ": ";
			std::cout << std::string(p[i] * nstars / nrolls, '*') << std::endl;
		}

		RandomEngineFactory::returnEngine(std::move(generator_ptr));
	}

	//exp dist plot hist
	void exponentialdist_hist_example() {
		const int nrolls = 8000;  // number of experiments
		const int nstars = 100;    // maximum number of stars to distribute
		const int nintervals = 10; // number of intervals

		auto generator_ptr = RandomEngineFactory::requestEngine();
		std::exponential_distribution<double> distribution(3.5);

		int p[nintervals] = {};

		for (int i = 0; i < nrolls; ++i) {
			double number = distribution(*generator_ptr);
			if (number < 1.0) ++p[int(nintervals*number)];
		}

		std::cout << "exponential_distribution (3.5):" << std::endl;
		std::cout << std::fixed; std::cout.precision(1);

		for (int i = 0; i < nintervals; ++i) {
			std::cout << float(i) / nintervals << "-" << float(i + 1) / nintervals << ": ";
			std::cout << std::string(p[i] * nstars / nrolls, '*') << std::endl;
		}

		RandomEngineFactory::returnEngine(std::move(generator_ptr));
	}
}