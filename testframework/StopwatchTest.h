#pragma once
#include "TestFramework.h"
#include <thread>

#include "../utility/Stopwatch.h"
#include "../utility/utilityfunctions.h"

class StopwatchTest :
	public TestFramework
{
public:

	StopwatchTest() : TestFramework("Stopwatch test") { }
	~StopwatchTest() = default;

	void operator()(void) { 
		//test(); 
		benchmark(); //mean +- standard deviation. Max and min for given routine.
	}
	void benchmark() {  
		auto ptr = std::bind( &StopwatchTest::test, this );
		m_tot_time = time_void<decltype(ptr)>(ptr);
		m_output_string << "completed in " << m_tot_time << " [s]" << std::endl;
	}
	void test() { stopwatch_test(); }

private:
	void stopwatch_test()
	{
		using namespace std::chrono_literals;
		Stopwatch<std::milli> sw;
		sw.getLapTime();
		auto start = std::chrono::high_resolution_clock::now();
		std::this_thread::sleep_for(100ms);
		auto end = std::chrono::high_resolution_clock::now();

		double stopwatchTime = sw.getLapTime();

		std::chrono::duration<double, std::milli> elapsed = end - start;
		m_output_string << "Waited " << elapsed.count() << " ms, " << " Stopwatch lap time=" << stopwatchTime << std::endl;

		const double absoluteErrorThreshold = 0.02; //20 micro sec
		if (std::abs(elapsed.count() - stopwatchTime) < absoluteErrorThreshold) {
			m_output_string << "Stopwatch test passed" << std::endl;
			m_success = true;
		}
		else {
			m_output_string << "Stopwatch test failed" << std::endl;
			m_success = false;
		}
	}
	double m_tot_time;
};

