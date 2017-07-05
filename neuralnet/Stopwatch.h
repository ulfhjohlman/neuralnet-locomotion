#pragma once
#include <ctime>
#include <chrono>
#include <string>

template< typename unit = std::ratio< 1i64, 1i64 > >
class Stopwatch final
{
public:
	Stopwatch() {
		m_end = std::chrono::high_resolution_clock::now();
		m_start = m_end;
	}
	~Stopwatch(){ }

	/// <summary>
	/// Get duration in specified unit(default seconds) since last call.
	/// </summary>
	/// <returns></returns>
	double getLapTime() {
		m_end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, unit> elapsed_time = m_end - m_start;
		m_start = m_end;

		return elapsed_time.count();
	}
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> m_start, m_end;
};
