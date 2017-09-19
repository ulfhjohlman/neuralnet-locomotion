#pragma once
#include<exception>
#include "../lib/tinyxml2/tinyxml2.h"
class XMLException : public std::runtime_error
{
public:
	XMLException() = delete;
	XMLException(tinyxml2::XMLError e) : std::runtime_error("A tinyxml2 exception occured."),  m_error(e) {}
	XMLException(const char* what) : std::runtime_error(what) {}
	XMLException(const char* what, tinyxml2::XMLError e) : std::runtime_error(what), m_error(e) {}
	virtual ~XMLException() _NOEXCEPT= default;
private:
	tinyxml2::XMLError m_error;
};
