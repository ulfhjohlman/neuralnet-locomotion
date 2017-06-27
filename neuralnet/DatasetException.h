#pragma once
#include<exception>

class DatasetException : public std::exception
{
public:
	DatasetException() : std::exception("A neural net exception occured.")  { }
	DatasetException(const char* what) : std::exception(what) { }
	virtual ~DatasetException(){ }
private:
};

