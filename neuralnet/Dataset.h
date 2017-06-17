#pragma once
#include "../lib/tinyxml2/tinyxml2.h"
#include <string>
#include <iostream>

#ifndef XMLCheckResult
#define XMLCheckResult(a_eResult) if (a_eResult != XML_SUCCESS) { printf("Error: %i\n", a_eResult); return a_eResult; }
#endif

class Dataset
{
public:
	Dataset() : m_doc(0), m_pRoot(0) {
		m_pRoot = m_doc.NewElement("Root");
		m_doc.InsertFirstChild(m_pRoot);

		tinyxml2::XMLElement * pElement = m_doc.NewElement("IntValue");

		pElement->SetText(10);
		m_pRoot->InsertEndChild(pElement);

		pElement = m_doc.NewElement("FloatValue");
		pElement->SetText(0.5f);
		m_pRoot->InsertEndChild(pElement);

		pElement = m_doc.NewElement("Date");
		pElement->SetAttribute("day", 26);
		pElement->SetAttribute("month", "April");
		pElement->SetAttribute("year", 2014);
		pElement->SetAttribute("dateFormat", "26/04/2014");
		m_pRoot->InsertEndChild(pElement);
		std::cout << pElement->Attribute("year") << std::endl;

		m_doc.InsertEndChild(m_pRoot);
		m_doc.SaveFile("poop.xml", false);
	}

	virtual ~Dataset() { }

	//virtual void loadFromFile(const char* name);
protected:
	tinyxml2::XMLDocument m_doc;
	tinyxml2::XMLNode* m_pRoot;
private:
};
