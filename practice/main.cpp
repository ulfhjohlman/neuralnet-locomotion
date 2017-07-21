#include <iostream>
#include <vector>
#include <memory>
#include <exception>
#include <type_traits>
#include <thread>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <limits>
#include <iomanip>
#include <random>
#include <functional>
#include <stdexcept>

#include "MemoryCapacity.h"
#include "Memory.h"
#include "LinkedList.h"

#include "../utility/utilityfunctions.h"
#include "../utility/Stopwatch.h"

#include "../lib/Eigen/dense"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <malloc.h>

void memory_test();
void threading_test();
void linkedlist_test();


void avx_test()
{
	//-----------------------------------------------------
	/* Initialize the two argument vectors */
	__m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
	__m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 17.0); 
	//ps = single precision, pd = double...

	/* Compute the difference between the two vectors */
	__m256 result = _mm256_sub_ps(evens, odds); //[-1,1,1,...]
	//Note that order is reversed after operation. i.e 
	//16 - 17 = -1 is first element

	/* Display the elements of the result vector */
	float* f = (float*)&result;
	printf("%f %f %f %f %f %f %f %f\n",
		f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
	//-----------------------------------------------------


	size_t alignment = 256;
	float* aligned_floats = (float*)_aligned_malloc( 32 , alignment); //8(floats) * 4(bytes per float) = 32(bytes), 256(bit) % 32(bit) = 0; 
	//This ptr should have a memory address that is a multiple of 256 bits.
	//This is also a then a multiple of 32 bits(float size), which is the size of each 
	//element in the 256 bit register. 

	if (aligned_floats == NULL){ printf_s("Error allocation aligned memory.");  }
	if (((unsigned long long)aligned_floats % alignment) == 0)
		printf_s("This pointer, %p, is aligned on %zu\n",
			aligned_floats, alignment);
	else
		printf_s("This pointer, %p, is not aligned on %zu\n",
			aligned_floats, alignment);

	for (int i = 0; i < 8; i++) {
		aligned_floats[i] = 1.0f + i;
		printf("%f ", aligned_floats[i]);
	}
	std::cout << std::endl;

	__m256 vec1 = _mm256_load_ps(aligned_floats); //load aligned
	__m256 vec2 = _mm256_set1_ps(1.05f);
	__m256 vec3 = _mm256_sub_ps(vec1, vec2);
	f = (float*)&vec3;
	printf("%f %f %f %f %f %f %f %f\n",
		f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);

	if (aligned_floats)
		_aligned_free(aligned_floats);
}

template<typename T>
void random_test() {
	// obtain a seed from the system clock:
	unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::mt19937_64 mt64_engine(seed);
	std::ranlux24 engine_ranlux(seed); //ranlux48 does not work!
	std::minstd_rand basic_rnd(seed);


	auto print_vector = [](const std::vector<T>& print_this) {
		std::for_each(print_this.begin(), print_this.end(),
			[](T element) { std::cout << element << std::endl; });
	};

	const int N = 1e3;
	std::vector<T> numbers(N);

	Stopwatch<std::milli> timer;
	std::generate(numbers.begin(), numbers.end(), [] { return std::sin(std::rand()); }); // warm up
																						 
	timer.getLapTime();
	std::generate(numbers.begin(), numbers.end(), [&engine_ranlux] {return std::generate_canonical<T, std::numeric_limits<T>::digits>(engine_ranlux); });
	double ranlux_time = timer.getLapTime();

	std::generate(numbers.begin(), numbers.end(), [&mt64_engine] {return std::generate_canonical<T, std::numeric_limits<T>::digits>(mt64_engine); });
	double mt64_time = timer.getLapTime();

	std::generate(numbers.begin(), numbers.end(), [&basic_rnd] {return std::generate_canonical<T, std::numeric_limits<T>::digits>(basic_rnd); });
	double minstd_time = timer.getLapTime();

	std::generate(numbers.begin(), numbers.end(), [] { return (T)std::rand() / (T)RAND_MAX; });
	double stdrand_time = timer.getLapTime();

	printf("ranlux24: %f [ms] \nmt64: %f [ms]\nminstd: %f [ms]\nstdrand: %f [ms]\n", ranlux_time, mt64_time, minstd_time, stdrand_time);
}

int main() {
	avx_test();

	std::cout << "double random test: " << std::endl;
	random_test<double>();
	std::cout << std::endl;

	std::cout << "float random test: " << std::endl;
	random_test<float>();
	std::cout << std::endl;
	
	//linkedlist_test();
	//threading_test();

	std::cout << "Done. Press any key to exit.\n";
	std::cin.get();
}

void memory_test()
{
	using namespace std::chrono_literals;
	MemoryCapacity::print();

	std::vector<std::unique_ptr< char >> chunks;

	auto allocate_chunk = [](size_t size) { return new char[size]; }; //why not

	try {
		while (true) {
			std::this_thread::sleep_for(50ms);
			chunks.push_back(std::unique_ptr<char>(allocate_chunk(MemoryCapacity::MB * 100))); //allocate 100mbchunks

			auto memoryleft = MemoryCapacity::getFreeRam();
			auto virtualMemoryLeft = MemoryCapacity::getFreeVirtualMemory();
			//std::cout << memoryleft << "+" << " ~0 " << " Bytes\n";

			if (memoryleft < MemoryCapacity::GB) {
				//Try allocate that gigabyte, can still throw bad_alloc.
				auto ptr = std::unique_ptr<char>(Memory::requestAllocation<char>(MemoryCapacity::GB));
				if (ptr == nullptr) //Is not true if OS can write to pagefile, i.e OS will make room for ptr
					std::cerr << "allocation request denied.\n";

				chunks.push_back(std::move(ptr)); //Push for the sake of it. Move ownership of data in unique pointer.
				throw std::exception("Less than 1 GB free ram left, buy or download some more.");
			}
		}
	}
	catch (std::bad_alloc& e) {
		std::cerr << e.what() << std::endl;
	}
	catch (std::exception& e) {
		std::cerr << e.what() << std::endl;
	}

	MemoryCapacity::print();
}


void threading_test()
{
	std::cout.sync_with_stdio(true);
	std::thread t1(memory_test);

	std::cout << "" << " From pool\n";
	if (t1.joinable())
		t1.join();
}

void linkedlist_test()
{
	std::vector<float> x(500);
	float x_values = -2;
	std::generate(x.begin(), x.end(), [=, &x_values] { return x_values += 0.01f; });

	std::vector<float> y(500);
	std::fill(y.begin(), y.end(), 0);
	int i = { 0 };
	std::generate(y.begin(), y.end(), [=, &x, &i] { return std::sin(x[i++]); }); //awkward
	std::transform(x.begin(), x.end(), y.begin(), [](float x) { return std::cos(x); });

	LinkedList<float>* p_head = new LinkedList<float>(5.0f);
	p_head->add(new LinkedList<float>(1.0f));
	p_head->add(new LinkedList<float>(3.0f));
	p_head->add(new LinkedList<float>(4.0f));
	(*p_head)[50] = 100;
	LinkedList<float> newcopy(1.0f);
	newcopy = *p_head;
	LinkedList<float> anothercopy(std::move(newcopy)); //invoke move constructor

	p_head->print();
	std::cout << std::endl;
	anothercopy.add(p_head); //Gonna be some weird ass add.

	p_head->print();
	std::cout << std::endl;
	newcopy.print();
	std::cout << std::endl;
	anothercopy.print();

	//delete p_head; //Memory get deleted by anothercopy destructor

	std::cout << "linked list done, press any key.";
	std::cin.get();
}
