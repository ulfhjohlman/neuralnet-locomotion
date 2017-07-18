#include "stdafx.h"
#include "Dataset.h"



void Dataset::createDataset(const char* name)
{
	m_doc.insertNewRoot(name);
	m_doc.insertDate();
}

void Dataset::setDescription(const char* dataset_description)
{
	m_doc.insertNewElement<const char*>("description", dataset_description);
}

void Dataset::setInputInfo(int number_of_variables, int number_of_points)
{
	m_input_info.first = number_of_variables;
	m_input_info.second = number_of_points;
	m_doc.insertNewElements<int>("input", std::vector<int>{number_of_variables, number_of_points});
}

void Dataset::setOutputInfo(int number_of_variables, int number_of_points)
{
	m_output_info.first = number_of_variables;
	m_output_info.second = number_of_points;
	m_doc.insertNewElements<int>("output", std::vector<int>{number_of_variables, number_of_points});
}

 void Dataset::setInputData(const char * data, const char * type) {
	if (type == "double" || type == "float" || type == "int") {
		m_doc.insertData("input data", data);
		m_doc.insertAttribute<const char*>("typename", type);
	}
	else
		throw DatasetException("type not available");
}

 void Dataset::setResultData(const char * data, const char * type) {
	if (type == "double" || type == "float" || type == "int") {
		m_doc.insertData("output data", data);
		m_doc.insertAttribute<const char*>("typename", type);
	}
	else
		throw DatasetException("type not available");
}

 void Dataset::save(const char * filename) {
	m_doc.saveToFile(filename);
}

 void Dataset::load(const char * filename) {
	m_doc.loadFromFile(filename);
}

 void Dataset::print() const {
	m_doc.print();
}
