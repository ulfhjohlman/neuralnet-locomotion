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
class XMLWrapper
{
public:
	XMLWrapper() : m_doc(false), m_rootNode(nullptr), m_currentElement(nullptr) { }

	virtual ~XMLWrapper() = default;

	virtual void saveToFile(const char* name, bool compact = false) {
		tinyxml2::XMLError error = m_doc.SaveFile(name, compact);
		XMLCheckResult(error);
	}
	virtual void loadFromFile(const char* name) {
		tinyxml2::XMLError error = m_doc.LoadFile(name);
		XMLCheckResult(error);

		m_rootNode = nullptr;
		m_rootNode = m_doc.FirstChild();
		if (m_rootNode == nullptr)
			throw XMLException("No elements in xml file found.", tinyxml2::XML_ERROR_FILE_READ_ERROR);
	}

	virtual void clearDocument() {
		m_doc.DeleteChildren();
		m_rootNode = nullptr;
		m_currentElement = nullptr;
	}

	/// <summary>
	/// Set the first node in the xml document. Only renames if root already present.
	/// </summary>
	/// <param name="documentRoot"></param>
	virtual void insertNewRoot(const char* documentRoot = "Root") {
		if (m_rootNode == nullptr)
			m_rootNode = m_doc.NewElement(documentRoot);
		else
			m_rootNode->SetValue(documentRoot);
		m_doc.InsertFirstChild(m_rootNode);
	}
	virtual void insertNewNode(const char* name) {
		tinyxml2::XMLElement* node = m_doc.NewElement(name);
		m_doc.InsertEndChild(node);
	}

	/// <summary>
	/// Can not select with duplicate names.
	/// </summary>
	/// <param name="name"></param>
	virtual void selectRootNode(const char* name) {
		m_rootNode = m_doc.FirstChildElement(name);
		checkRootNode();
	}

	virtual void selectCurrentElement(const char* name) {
		checkRootNode();
		m_currentElement = m_rootNode->FirstChildElement(name);
		checkCurrentElement();
	}

	/// <summary>
	/// Get the number of items in current element
	/// </summary>
	/// <returns></returns>
	virtual unsigned int getNumberOfItems() {
		checkCurrentElement();
		unsigned int ret = 0;
		tinyxml2::XMLError error = m_currentElement->QueryUnsignedAttribute("itemCount", &ret);
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

	/// <summary>
	/// Inserts a new element of type valid in tinyxml2. That is, uint float double bool char* etc
	/// </summary>
	/// <param name="elementName"></param>
	/// <param name="textValue"></param>
	template<typename T>
	void insertNewElement(const char* elementName, T textValue) {
		checkRootNode();

		m_currentElement = m_doc.NewElement(elementName);
		m_currentElement->SetText(textValue);

		m_rootNode->InsertEndChild(m_currentElement);
	}

	template<typename T>
	void insertNewElements(const char* listName, const std::vector<T>& elements) {
		checkRootNode();

		const size_t size = elements.size();
		if (size < 1) throw std::runtime_error("Empty vector.");

		m_currentElement = m_doc.NewElement(listName);

		for (const auto& element : elements) {
			tinyxml2::XMLElement* item = m_doc.NewElement("Item");
			item->SetText(element);

			m_currentElement->InsertEndChild(item);
		}

#ifdef _DEBUG
		assert(elements.size() < UINT32_MAX); //one massive vector
#endif // _DEBUG
		m_currentElement->SetAttribute("itemCount", static_cast<unsigned int>(size));
		m_rootNode->InsertEndChild(m_currentElement);
	}

	template<typename T>
	void insertData(const char* dataName, const std::vector<T>& data) {
		checkRootNode();

#ifdef _DEBUG
		assert(data.size() < UINT32_MAX);
#endif // _DEBUG
		const unsigned int size = data.size();
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

	template<typename T>
	void getElement(const char* elementName, T& element) {
		checkRootNode();
		m_currentElement = m_rootNode->FirstChildElement(elementName);
		checkCurrentElement();

		throw XMLException("Fuck this.");
	}

	void getElements(const char* elementName, std::vector<int>& elements) {
		checkRootNode();
		m_currentElement = m_rootNode->FirstChildElement(elementName);
		checkCurrentElement();

		int group_size;
		XMLCheckResult(m_currentElement->QueryIntAttribute("itemCount",&group_size));
		m_currentElement = m_currentElement->FirstChildElement("Item");
		for (size_t i = 0; i < group_size; i++) {
			int out;
			checkCurrentElement();

			XMLCheckResult(m_currentElement->QueryIntText(&out));

			elements.push_back(out);
			m_currentElement = m_currentElement->NextSiblingElement();
		}
	}
	void getElements(const char* elementName, std::vector<float>& elements) {
		checkRootNode();
		m_currentElement = m_rootNode->FirstChildElement(elementName);
		checkCurrentElement();

		int group_size;
		XMLCheckResult(m_currentElement->QueryIntAttribute("itemCount", &group_size));
		m_currentElement = m_currentElement->FirstChildElement("Item");
		for (size_t i = 0; i < group_size; i++) {
			float out;
			checkCurrentElement();

			XMLCheckResult(m_currentElement->QueryFloatText(&out));

			elements.push_back(out);
			m_currentElement = m_currentElement->NextSiblingElement();
		}
	}
	void getElements(const char* elementName, std::vector<double>& elements) {
		checkRootNode();
		m_currentElement = m_rootNode->FirstChildElement(elementName);
		checkCurrentElement();

		int group_size;
		XMLCheckResult(m_currentElement->QueryIntAttribute("itemCount", &group_size));
		m_currentElement = m_currentElement->FirstChildElement("Item");
		for (size_t i = 0; i < group_size; i++) {
			double out;
			checkCurrentElement();

			XMLCheckResult(m_currentElement->QueryDoubleText(&out));

			elements.push_back(out);
			m_currentElement = m_currentElement->NextSiblingElement();
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
