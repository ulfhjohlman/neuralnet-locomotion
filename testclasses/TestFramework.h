#pragma once
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

#include "../utility/XMLWrapper.h"

class TestFramework
{
public:
	TestFramework() : m_success(true), m_name("Test class") { }
	TestFramework(const char* name) : m_success(true), m_name(name) { }
	virtual ~TestFramework() = default;

	virtual void setName(const char* name) {
		m_name = name;
	}
	//virtual void operator()() = 0;
	virtual void benchmark() = 0;
	virtual void test() = 0;
	virtual void print() const {
#ifdef _DEBUG
		std::cout << "Debug build: ";
#else
		std::cout << "Release build: ";
#endif // DEBUG
	std::cout << m_name << std::endl << m_output_string.str() << std::endl;
	}
	virtual void save() const {
		try {
			XMLWrapper doc;
			std::string filename;
#ifdef _DEBUG
			filename = m_name + "-debug";
#else
			filename = m_name + "-release";
#endif // _DEBUG

			doc.insertNewRoot(filename.c_str());
			doc.insertDate();
			doc.insertNewElement("successful", m_success);
			doc.insertNewElement("mean", m_mean);
			doc.insertNewElement("min", m_min);
			doc.insertNewElement("max", m_max);
			doc.insertNewElement("std", standard_diviation);
			doc.saveToFile(filename.c_str());
		}
		catch (XMLException e) {
			std::cerr << e.what() << std::endl;
		}
		catch (std::exception e){
			std::cerr << e.what() << std::endl;
		}
	}
	bool passed() const { return m_success; };

protected:
	std::ostringstream m_output_string;
	std::string m_name;
	bool m_success;
	double m_mean, m_min, m_max, standard_diviation;
private:
};