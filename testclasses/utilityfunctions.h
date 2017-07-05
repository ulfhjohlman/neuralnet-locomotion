#pragma once
#include <functional>
#include <algorithm>
#include <iterator>
#include <thread>
#include <future>
#include <string>
#include <type_traits>

template<typename Iterator, typename Function, unsigned long min_per_thread = 100>
void parallel_for_each(Iterator first, Iterator last, Function f) {
	const unsigned long length = std::distance(first, last);

	/*if (length == 0)
		return;

	if (length < 2UL * min_per_thread)
		return parallel_for_each(first, last, f);
	else {
		Iterator mid = first + length / 2;
		std::future<void> fi;
	}*/
}

template<unsigned int i, typename Function>
struct Loop
{
	static inline void unroll(Function inloop) {
		Loop< i - 1, Function >::unroll(inloop);
		inloop();
	}
};

template<typename Function>
struct Loop < -1, Function >
{
	static inline void unroll(Function inloop) { }
};

template <typename Function, typename ...Args>
auto time(Function f, Args && ...args)
{
	static_assert( !std::is_void<decltype(f(args...))>::value, "Call time_void if return type is void!");

	auto start =  high_resolution_clock::now();
	auto result = f(std::forward<Args>(args)...);
	auto end =    high_resolution_clock::now();
	duration<double> elapsed_seconds = end - start;

	return std::pair<decltype(f(args...)), double>{ result, elapsed_seconds.count() };
}

template <typename Function, typename ...Args>
auto time_void(Function f,  Args && ...args) 
{
	static_assert( std::is_void<decltype(f(args...))>::value, "Call time if return type is not void!");
	using std::chrono;


	auto start = high_resolution_clock::now();
	f(std::forward<Args>(args)...);
	auto end = high_resolution_clock::now();
	duration<double> elapsed_seconds = end - start;

	return elapsed_seconds.count();
}
