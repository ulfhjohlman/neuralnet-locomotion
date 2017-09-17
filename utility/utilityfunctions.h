#pragma once
#include <string>
#include <chrono>
#include <ctime>
#include <functional>
#include <algorithm>
#include <iterator>
#include <thread>
#include <future>
#include <string>
#include <type_traits>


#include "ThreadPoolv2.h"
template<typename Iterator, typename Function >
void parallel_for_each(Iterator first, Iterator last, Function f, size_t block_size = 100) {
	const size_t length = std::distance(first, last);

	if (length == 0)
		return;
	ThreadPool pool;
	size_t number_items = length / block_size; //integer math rounds down
	
	Iterator block_start = first;
	Iterator block_end = block_start;
	
	for (size_t i = 0; i < number_items; i++) {
		std::advance(block_end, block_size); 
		auto wrap_f = std::bind( std::for_each<decltype(block_start), decltype(f)>, block_start, block_end, f );// f : [ block_start, block_start + block_size ] -> f(x) 
		pool.addWork( std::move(wrap_f) );
		block_start = block_end;
	}
	if (block_end != last) {
		block_end = last;
		auto wrap_f = std::bind(std::for_each<decltype(block_start), decltype(f)>, block_start, block_end, f);// f : [ block_start, block_start + block_size ] -> f(x) 
		pool.addWork( std::move(wrap_f));
	}

	pool.help();
	while (!pool.isDone()){ }
}

/// <summary>
/// <code> 
///auto l = [](int a, int b) { std::cout << a + b << std::endl; };
///Loop 3, decltype(l), int, int ::unroll(l, 2, 3);
/// </code>
/// </summary>
template<unsigned int i, typename Function, typename ...Args>
struct Loop
{
	static inline void unroll(Function inloop, Args && ...args) {
		Loop< i - 1, Function, Args... >::unroll(inloop, std::forward<Args>(args)...);
		inloop(std::forward<Args>(args)...);
	}
};

template<typename Function, typename ...Args>
struct Loop < 0, Function, Args... >
{
	static inline void unroll(Function inloop, Args&&... args ) { }
};

template <typename Function, typename ...Args>
auto time_template(Function f, Args && ...args)
{
	static_assert( !std::is_void<decltype(f(args...))>::value, "Call time_void if return type is void!");
	using namespace std::chrono;

	auto start =  high_resolution_clock::now();
	auto result = f(std::forward<Args>(args)...);
	auto end =    high_resolution_clock::now();
	duration<double> elapsed_seconds = end - start;

	return std::pair<decltype(f(args...)), double>{ result, elapsed_seconds.count() };
}

template <typename Function, typename ...Args>
double time_void(Function f,  Args && ...args) 
{
	static_assert( std::is_void<decltype(f(args...))>::value, "Call time if return type is not void!");
	using namespace std::chrono;

	auto start = high_resolution_clock::now();
	f(std::forward<Args>(args)...);
	auto end = high_resolution_clock::now();
	duration<double> elapsed_seconds = end - start;

	return elapsed_seconds.count();
}