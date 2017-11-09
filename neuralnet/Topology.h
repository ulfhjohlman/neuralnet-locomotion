#pragma once

#include "NeuralNet.h"

/// <summary>
/// Defines a general topology, empty for now.
/// </summary>
class Topology 
{
public:
	Topology() = default;
	virtual ~Topology() = default;

	virtual void save(const char* file_name) {
		m_document.print();
		m_document.save(file_name);
	}
	virtual void load(const char* file_name) {
		m_document.load(file_name);
		m_document.print();
	}
protected:
	XMLFile m_document;
private:
};
