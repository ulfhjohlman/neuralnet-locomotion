#pragma once
#include<stdexcept>

class DatasetException : public std::runtime_error
{
public:
	DatasetException() : std::runtime_error("A neural net exception occurred.")  { }
	DatasetException(const char* what) : std::runtime_error(what) { }
	virtual ~DatasetException() = default;
private:
	const char* str;
};
