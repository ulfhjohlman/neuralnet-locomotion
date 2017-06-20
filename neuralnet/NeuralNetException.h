#pragma once
#include <exception>

class NeuralNetException : public std::exception
{
public:
	NeuralNetException() : std::exception("A neural net exception occured.") { }
	NeuralNetException(const char* what) : std::exception(what) {}
	virtual ~NeuralNetException() = default;
private:
};
