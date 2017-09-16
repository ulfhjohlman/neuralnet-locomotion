#pragma once
#include <stdexcept>

class NeuralNetException : public std::runtime_error
{
public:
	NeuralNetException() : std::runtime_error("A neural net exception occurred.") { }
	NeuralNetException(const char* what) : std::runtime_error(what) {}
	virtual ~NeuralNetException() = default;
private:
};
