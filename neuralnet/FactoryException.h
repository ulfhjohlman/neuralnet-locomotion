#pragma once
#include <stdexcept>

class FactoryException : public std::runtime_error
{
public:
	FactoryException() : std::runtime_error("A factory exception occurred.") { }
	FactoryException(const char* what) : std::runtime_error(what) {}
	virtual ~FactoryException() = default;
private:
};
