#pragma once
#include "../lib/tinyxml2/tinyxml2.h"
#include "DatasetException.h"

#include "../utility/XMLFile.h"
#include "../utility/XMLException.h"
#include "../utility/DataPrinter.h"
#include "../utility/utilityfunctions.h"

#include <string>
#include <iostream>
#include <vector>


/// <summary>
/// A class for wrapping tinyxml2 features into a object that can store data related to a dataset. 
/// The dataset throws exceptions in both debug and release build. So it's meant to handle errors regardless
/// of operation. 
/// 
/// Is not thread safe. Locking might be implemented for client use in two specific functions.
/// Can not handle large sets(> memory) of data. And does not buffer anything.
/// 
/// Class far from done or properly specialized!
/// 
/// Fuck this unnecessary shit
/// </summary>
class Dataset
{
public:
	Dataset() = default;
	~Dataset() = default;

	virtual void createDataset(const char* name);
	virtual void setDescription(const char* dataset_description);

	virtual void setInputInfo(int number_of_variables, int number_of_points);
	virtual void setOutputInfo(int number_of_variables, int number_of_points);

	virtual void setInputData(const char* data, const char* type);
	virtual void setResultData(const char* data, const char* type);

	virtual void save(const char* filename);
	virtual void load(const char* filename);

	virtual void print() const;

public:
	void getInputInfo() {
		std::vector<int> tmp;
		m_doc.getElements("input", tmp);
		m_input_info.first = tmp[0];
		m_input_info.second = tmp[1];
	}
	void getOutputInfo() {
		std::vector<int> tmp;
		m_doc.getElements("output", tmp);
		m_output_info.first = tmp[0];
		m_output_info.second = tmp[1];
	}
	template<typename T>
	void getInputData() {
		std::vector<T> tmp;
		m_doc.getElements("input data", tmp);
	}
	template<typename T>
	void getOutputData() {
		std::vector<T> tmp;
		m_doc.getElements("output data", tmp);

		//for_each(tmp.begin(), tmp.end(), [](T a) { std::cout << a " "; });
	}


private:
	std::pair<int, int> m_input_info, m_output_info;
	XMLFile m_doc;
};
