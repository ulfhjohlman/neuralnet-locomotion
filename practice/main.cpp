#include <iostream>
#include <vector>
#include <memory>
#include <exception>
#include <type_traits>
#include <thread>
#include <chrono>
#include <algorithm>
#include <numeric>

#include "MemoryCapacity.h"
#include "Memory.h"
#include "LinkedList.h"

#include "../testclasses/ThreadPool.h"
#include "../testclasses/ThreadsafeQueue.h"
#include "../testclasses/FunctionWrapper.h"
#include "../testclasses/utilityfunctions.h"


void memory_test();
void threading_test();



template<typename Iterator, typename T>
struct accumulate_block
{
	accumulate_block(Iterator first_, Iterator last_): first(first_), last(last_) { }
	Iterator first, last;

	T operator()()
	{
		return std::accumulate(first, last, T());
	}
};

template<typename Iterator, typename T>
T parallel_accumulate(Iterator first, Iterator last, T init)
{
	unsigned long const length = std::distance(first, last);
	if (!length)
		return init;
	unsigned long const block_size = 25;
	unsigned long const num_blocks = (length + block_size - 1) / block_size;
	std::vector<std::future<T> > futures(num_blocks - 1);
	ThreadPool pool;
	Iterator block_start = first;
	for (unsigned long i = 0; i < (num_blocks - 1); ++i)
	{
		Iterator block_end = block_start;
		std::advance(block_end, block_size);
		accumulate_block<Iterator, T> block(block_start, block_end);
		futures[i] = pool.submit(block);
		block_start = block_end;
	}
	accumulate_block<Iterator, T> block_last(block_start, last);
	T last_result = block_last();
	T result = init;
	for (unsigned long i = 0; i < (num_blocks - 1); ++i)
	{
		result += futures[i].get();
	}
	result += last_result;
	return result;
}


int main() {
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


	newcopy[10] = 59;

	p_head->print();
	std::cout << std::endl;
	newcopy.print();

	delete p_head;

	//float sum = parallel_accumulate(x.begin(), x.end(), 0.0f);
	//threading_test();
	std::cout << "practice done.";
	std::cin.get();
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
			std::cout << memoryleft << "+" << virtualMemoryLeft << " B\n";

			if (memoryleft < MemoryCapacity::GB) {
				//Try allocate that gigabyte, can still throw bad_alloc.
				auto ptr = std::unique_ptr<char>(Memory::requestAllocation<char>(MemoryCapacity::GB));
				if (ptr == nullptr) //Is not true if OS can write to pagefile
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

	//auto ptr = [](float x) { return std::sin(x) + std::cos(x); };
	//ThreadPool pool;
	//auto fut = pool.submit([]() { return std::sin(10) + std::cos(10); });
	//auto answer = fut.get();

	std::cout << "" << " From pool\n";
	if (t1.joinable())
		t1.join();
}