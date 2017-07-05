#pragma once
#include <string>
#include <chrono>
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

/// <summary>
/// <code> 
///auto l = [](int a, int b) { std::cout << a + b << std::endl; };
///Loop<3, decltype(l), int, int>::unroll(l, 2, 3);
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
auto time(Function f, Args && ...args)
{
	static_assert( !std::is_void<decltype(f(args...))>::value, "Call time_void if return type is void!");
	using std::chrono;

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

/// <summary>
/// Get date and time from this computer. Not thread safe!
/// </summary>
/// <returns>string day/month/year hours:minutes:seconds</returns>
std::string getAbsoluteTime() {
	std::time_t date_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
	return std::ctime(&date_time); //Use strftime instead
}