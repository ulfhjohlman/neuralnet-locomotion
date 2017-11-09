#pragma once
#include "../lib/tinyxml2/tinyxml2.h"
#include "XMLException.h"
#include "Stopwatch.h"
#include "DataPrinter.h"


#include <string>
#include <iostream>
#include <vector>
#include <cassert> //assert
#include <iomanip> //std::tm
#include <ctime> //localtime
#include <stdexcept> //runtime_exception

/// <summary>
/// <para/>A class for wrapping tinyxml2 features into a object that can store data.
/// Throws exceptions in both debug and release build. So it's meant to handle errors regardless
/// of operation.
///
/// <para/>Is not thread safe. Locking might be implemented for client use in two specific functions.
/// Can not handle large sets(> memory) of data. And does not buffer anything.
///
/// <para/>Not completed.
/// </summary>
class XMLFile
{
public:
	XMLFile() : m_doc(false), m_rootNode(nullptr), m_currentElement(nullptr) { }

	virtual ~XMLFile() = default;

	virtual void save(const char* name, bool compact = false) {
		tinyxml2::XMLError error = m_doc.SaveFile(name, compact);
		XMLCheckResult(error);
	}
	virtual void load(const char* name) {
		if (m_rootNode) {
			this->clear();
			//throw XMLException("There is already a active document.", tinyxml2::XML_ERROR_FILE_READ_ERROR);
		}

		tinyxml2::XMLError error = m_doc.LoadFile(name);
		XMLCheckResult(error);

		m_rootNode = m_doc.FirstChild();
		m_currentElement = m_rootNode->ToElement();
		if (m_rootNode == nullptr)
			throw XMLException("No elements in xml file found.", tinyxml2::XML_ERROR_FILE_READ_ERROR);
	}

	virtual void clear() {
		m_doc.DeleteChildren();
		m_rootNode = nullptr;
		m_currentElement = nullptr;
	}

	/// <summary>
	/// insert the node in the xml document.
	/// </summary>
	/// <param name="documentRoot"></param>
	virtual void insert(const char* node_name) {
		if (m_rootNode == nullptr) {
			m_rootNode = m_doc.NewElement(node_name);
			m_doc.InsertFirstChild(m_rootNode);
			m_currentElement = m_rootNode->ToElement();
		}
		else {
			checkCurrentElement();
			tinyxml2::XMLElement* element = m_doc.NewElement(node_name);
			m_currentElement->InsertEndChild(element);
		}
	}

	/// <summary>
	/// Inserts a new element of type valid in tinyxml2. That is, uint float double bool char* etc
	/// </summary>
	/// <param name="elementName"></param>
	/// <param name="textValue"></param>
	template<typename T>
	void insert(const char* elementName, T textValue) {
		checkRootNode();
		checkCurrentElement();
		tinyxml2::XMLElement* element = m_doc.NewElement(elementName);
		element->SetText(textValue);

		m_currentElement->InsertEndChild(element);
	}

	virtual void select(const char* name) {
		checkRootNode();
		m_currentElement = m_currentElement->FirstChildElement(name);
		checkCurrentElement();
	}

	virtual void selectRoot() {
		checkRootNode();
		m_currentElement = m_rootNode->ToElement();
		checkCurrentElement();
	}

	/// <summary>
	/// Get the number of items in current element
	/// </summary>
	/// <returns></returns>
	virtual int getNumberOfItems() {
		checkCurrentElement();
	    int ret = 0;
		tinyxml2::XMLError error = m_currentElement->QueryIntAttribute("itemCount", &ret);
		XMLCheckResult(error, "Not able to get item count.");
		return ret;
	}

	/// <summary>
	/// Insert date attributes into a "Date" element.
	/// </summary>
	virtual void insertDate() {
		tinyxml2::XMLElement* date = m_doc.NewElement("Date");

//#pragma warning(disable:4996)
		auto t = std::time(nullptr);
		auto tm = *std::localtime(&t);
//#pragma  warning(default:4996)

		date->SetAttribute("day", tm.tm_mday);
		date->SetAttribute("month", tm.tm_mon + 1);
		date->SetAttribute("year", tm.tm_year + 1900);

		date->SetAttribute("hour", tm.tm_hour);
		date->SetAttribute("min", tm.tm_min);
		date->SetAttribute("second", tm.tm_sec);

		m_rootNode->InsertFirstChild(date);
	}

	template<typename T>
	void insertElements(const char* node_name, const std::vector<T>& elements, const char* item_name = "") {
		checkRootNode();
		checkCurrentElement();

		int size = static_cast<int>(elements.size());
		if (size < 1) throw std::runtime_error("Empty vector.");

		tinyxml2::XMLElement* new_element = m_doc.NewElement(node_name);

		for (const auto& element : elements) {
			tinyxml2::XMLElement* item = m_doc.NewElement(item_name);
			item->SetText(static_cast<T>(element));

			new_element->InsertEndChild(item);
		}

		new_element->SetAttribute("itemCount", size);
		m_currentElement->InsertEndChild(new_element);
		//Does not change current element
	}

	template<typename T>
	void insertData(const char* dataName, const std::vector<T>& data) {
		checkRootNode();

#ifdef _DEBUG
		assert(data.size() < UINT32_MAX);
#endif // _DEBUG
		int size = static_cast<int>(data.size());
		if (size < 1) throw std::runtime_error("Empty vector.");

		m_currentElement = m_doc.NewElement(dataName);
		m_dataprinter.write<T>(data);
		m_currentElement->SetText(m_dataprinter.getString().c_str());

		m_currentElement->SetAttribute("itemCount", size);
		m_rootNode->InsertEndChild(m_currentElement);
	}

	void insertData(const char* dataName, const char* data) {
		checkRootNode();

		m_currentElement = m_doc.NewElement(dataName);
		m_currentElement->SetText(data);

		m_rootNode->InsertEndChild(m_currentElement);
	}


	/// <summary>
	/// Inserts attribute at current selected element. Last modified or
	/// </summary>
	/// <param name="attributeName"></param>
	/// <param name="attribute"></param>
	template<typename T>
	void insertAttribute(const char* attributeName, T attribute) {
		checkCurrentElement();
		m_currentElement->SetAttribute(attributeName, attribute);
	}
	virtual void print() const {
		m_doc.Print();
	}

	void getAttribute(const char* name, int& a) {
		auto error = m_currentElement->QueryIntAttribute(name, &a);
		XMLCheckResult(error);
	}

	void getElement(const char* elementName, char data[]) {
		checkRootNode();
		checkCurrentElement();
		tinyxml2::XMLElement* element = m_currentElement; //hold previous
		m_currentElement = m_currentElement->FirstChildElement(elementName);
		checkCurrentElement();

		std::strcpy(data, m_currentElement->GetText());
		m_currentElement = element; //return
	}

	void getElements(const char* elementName, std::vector<int>& elements, const char* item_name= "") {
		checkRootNode();
		checkCurrentElement();
		tinyxml2::XMLElement* it = m_currentElement->FirstChildElement(elementName);
		if (it == nullptr)
			throw XMLException("not list.");

		int group_size;
		XMLCheckResult(it->QueryIntAttribute("itemCount", &group_size));
		it = it->FirstChildElement(item_name);
		elements.reserve(group_size);
		for (int i = 0; i < group_size; i++) {
			int out;

			XMLCheckResult(it->QueryIntText(&out));

			elements.push_back(out);
			it = it->NextSiblingElement();
		}
	}
	void getElements(const char* elementName, std::vector<float>& elements, const char* item_name = "") {
		checkRootNode();
		checkCurrentElement();
		tinyxml2::XMLElement* it = m_currentElement->FirstChildElement(elementName);
		if (it == nullptr)
			throw XMLException("not list.");

		int group_size;
		XMLCheckResult(it->QueryIntAttribute("itemCount", &group_size));
		it = it->FirstChildElement(item_name);
		elements.reserve(group_size);
		for (int i = 0; i < group_size; i++) {
			float out;

			XMLCheckResult(it->QueryFloatText(&out));

			elements.push_back(out);
			it = it->NextSiblingElement();
		}
	}
	void getElements(const char* elementName, std::vector<double>& elements, const char* item_name = "") {
		checkRootNode();
		checkCurrentElement();
		tinyxml2::XMLElement* it = m_currentElement->FirstChildElement(elementName);
		if (it == nullptr)
			throw XMLException("not list.");

		int group_size;
		XMLCheckResult(it->QueryIntAttribute("itemCount", &group_size));
		it = it->FirstChildElement(item_name);
		elements.reserve(group_size);
		for (int i = 0; i < group_size; i++) {
			double out;

			XMLCheckResult(it->QueryDoubleText(&out));

			elements.push_back(out);
			it = it->NextSiblingElement();
		}
	}




protected:
	tinyxml2::XMLDocument m_doc;
	tinyxml2::XMLNode* m_rootNode;
	tinyxml2::XMLElement* m_currentElement;

	DataPrinter m_dataprinter;

	/// <summary>
	/// Throws XMLException exception if some operation don't return XML_SUCCESS
	/// </summary>
	/// <param name="error"> A tinyxml2::error_code </param>
	void XMLCheckResult(tinyxml2::XMLError error) const {
		if (error != tinyxml2::XML_SUCCESS) {
			throw XMLException(error);
		}
	}
	void XMLCheckResult(tinyxml2::XMLError error, const char* errorMessage) const {
		if (error != tinyxml2::XML_SUCCESS) {
			throw XMLException(errorMessage, error);
		}
	}
	void checkRootNode() const {
		if (m_rootNode == nullptr)
			throw std::runtime_error("No root node present.");
	}
	void checkCurrentElement() const {
		if (m_currentElement == nullptr)
			throw std::runtime_error("No element selected.");
	}
};
