#pragma once
#include<exception>
#include "../lib/tinyxml2/tinyxml2.h"
class XMLException :
	public std::exception
{
public:
	XMLException() = delete;
	XMLException(tinyxml2::XMLError e) : std::exception("A tinyxml2 exception occured."),  m_error(e) {}
	XMLException(const char* what) = delete;
	XMLException(const char* what, tinyxml2::XMLError e) : std::exception(what), m_error(e) {}
	virtual ~XMLException() = default;
private:
	tinyxml2::XMLError m_error;
};

