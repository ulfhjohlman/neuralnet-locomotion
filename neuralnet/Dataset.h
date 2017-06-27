#pragma once
#include "../lib/tinyxml2/tinyxml2.h"
#include "DatasetException.h"
#include "XMLException.h"
#include "Stopwatch.h"
#include "DataPrinter.h"

#include <string>
#include <iostream>
#include <vector>
#include <map>

/// <summary>
/// A class for wrapping tinyxml2 features into a object that can store data related to a dataset. 
/// The dataset throws exceptions in both debug and release build. So it's meant to handle errors regardless
/// of operation. 
/// 
/// Is not threadsafe. Locking might be implemented for client use in two specific functions.
/// Can not handle large sets(> memory) of data. And does not buffer anything.
/// </summary>
class Dataset
{
public:
	Dataset() : m_doc(0), m_rootNode(0), m_currentElement(0) {
	}

	virtual ~Dataset() = default;

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
		if( m_rootNode == nullptr )
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
	virtual unsigned int getNumberOfItems() const {
		checkCurrentElement();
		tinyxml2::XMLAttribute* attribute = m_currentElement->FindAttribute("a ");
		return 1;
	}

	/// <summary>
	/// Is bugged. Reads out of bounds when day is single digit. Need to get pos of whitespace.
	/// </summary>
	virtual void insertDate() {
		tinyxml2::XMLElement* date = m_doc.NewElement("Date");
		std::string date_string = m_stopwatch.getAbsoluteTime();

		date->SetAttribute("day", date_string.substr(0, 3).c_str() );
		date->SetAttribute("month", date_string.substr(4, 3).c_str());

		date->SetAttribute("date", date_string.substr(8, 2).c_str() ); //This offset should sometimes be 1.
		date->SetAttribute("time", date_string.substr(11, 8).c_str());
		date->SetAttribute("year", date_string.substr(20,4).c_str() ); //Will not work

		m_rootNode->InsertFirstChild( date );
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

		const unsigned int size = elements.size();
		if (size < 1) throw DatasetException("Empty vector.");

		 m_currentElement = m_doc.NewElement(listName);

		for (const auto& element : elements){
			tinyxml2::XMLElement* item = m_doc.NewElement("Item");
			item->SetText(element);

			m_currentElement->InsertEndChild(item);
		}
		m_currentElement->SetAttribute("itemCount", size);
		m_rootNode->InsertEndChild(m_currentElement);
	}

	template<typename T>
	void insertData(const char* dataName, const std::vector<T>& data) {
		checkRootNode();

		const unsigned int size = data.size();
		if (size < 1) throw DatasetException("Empty vector.");

		m_currentElement = m_doc.NewElement(dataName);
		m_dataprinter.write(data);
		m_currentElement->setText( m_dataprinter.getString().c_str() );

		m_currentElement->SetAttribute("itemCount", size);
		m_rootNode->InsertEndChild(m_currentElement);
	}

	template<typename T>
	void insertData(const char* dataName, const char* data) {
		checkRootNode();

		m_currentElement = m_doc.NewElement(dataName);
		m_currentElement->setText(data);

		m_rootNode->InsertEndChild(m_currentElement);
	}

	
	/// <summary>
	/// Inserts attribute at current selected element. Last modified or 
	/// </summary>
	/// <param name="attributeName"></param>
	/// <param name="attribute"></param>
	template<typename T>
	void insertAttribute( const char* attributeName, T attribute ) {
		checkCurrentElement();
		m_currentElement->SetAttribute(attributeName, attribute);
	}
	virtual void print() const {
		m_doc.Print();
	}
protected:
	tinyxml2::XMLDocument m_doc;
	tinyxml2::XMLNode* m_rootNode;
	tinyxml2::XMLElement* m_currentElement;

	Stopwatch<> m_stopwatch;
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
			throw XMLException(errorMessage ,error);
		}
	}
	void checkRootNode() const {
		if (m_rootNode == nullptr)
			throw DatasetException("No root node present.");
	}
	void checkCurrentElement() const {
		if (m_currentElement == nullptr)
			throw DatasetException("No element selected.");
	}
};
