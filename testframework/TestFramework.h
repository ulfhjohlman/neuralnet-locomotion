#pragma once
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

#include "../utility/XMLWrapper.h"

class TestFramework
{
public:
	TestFramework();
	TestFramework(const char* name);
	virtual ~TestFramework() = default;

	virtual void setName(const char* name);
	//virtual void operator()() = 0;
	virtual void benchmark() = 0;
	virtual void test() = 0;
	virtual void print() const;
	virtual void save() const;
	bool passed() const;;

protected:
	std::ostringstream m_output_string;
	std::string m_name;
	bool m_success;
	double m_mean, m_min, m_max, standard_diviation;
private:
};