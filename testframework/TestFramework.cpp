#include "TestFramework.h"

TestFramework::TestFramework() : m_success(true), m_name("Test class")
{

}

TestFramework::TestFramework(const char* name) : m_success(true), m_name(name)
{

}

void TestFramework::setName(const char* name)
{
	m_name = name;
}

void TestFramework::print() const
{
#ifdef _DEBUG
	std::cout << "Debug build: ";
#else
	std::cout << "Release build: ";
#endif // DEBUG
	std::cout << m_name << std::endl << m_output_string.str() << std::endl;
}

void TestFramework::save() const
{
	try {
		XMLFile doc;
		std::string filename;
#ifdef _DEBUG
		filename = m_name + "-debug";
#else
		filename = m_name + "-release";
#endif // _DEBUG

		doc.insert(filename.c_str());
		doc.insertDate();
		doc.insert("successful", m_success);
		doc.insert("mean", m_mean);
		doc.insert("min", m_min);
		doc.insert("max", m_max);
		doc.insert("std", standard_diviation);
		doc.save(filename.c_str());
	}
	catch (XMLException e) {
		std::cerr << e.what() << std::endl;
	}
	catch (std::exception e) {
		std::cerr << e.what() << std::endl;
	}
}

bool TestFramework::passed() const
{
	return m_success;
}
