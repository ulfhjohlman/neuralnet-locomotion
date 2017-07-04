#pragma once
#include <stdexcept>
#include <iostream>

template<typename T>
class LinkedList
{
public:
	LinkedList();
	LinkedList(T t_);
	
	~LinkedList();

	void add(LinkedList<T>* node);
	const LinkedList<T>* getNext() const;
	void print();

	T& operator[](const size_t index);
	LinkedList<T>& operator=(const LinkedList<T>& rhs);
protected:
	
private:
	T m_value;
	LinkedList<T>* m_next;
};

template<typename T>
LinkedList<T>& LinkedList<T>::operator=(const LinkedList<T>& rhs) //FUUUUCK forgot &
{
	std::cout << "HEYHEY\n";
	if (this == &rhs)
		throw std::exception("NOPE");

	this->m_value = rhs.m_value;
	//const LinkedList<T>* it = &rhs;
	//while (it->getNext()) {
	//	this->add(new LinkedList<T>(it->getNext()->m_value));
	//	it = it->getNext();
	//}

	return *this;
}

template<typename T>
LinkedList<T>::LinkedList(T t_ ) : LinkedList<T>()
{
	m_value = t_;
}

template<typename T>
T& LinkedList<T>::operator[](const size_t index)
{
	LinkedList<T>* it = this;
	for(size_t i = 0; i < index; i++)
		if (it->m_next)
			it = it->m_next;
	
	return it->m_value;
}

template<typename T>
void LinkedList<T>::print()
{
	std::cout << m_value << std::endl;
	if (m_next)
		m_next->print();
}

template<typename T>
const LinkedList<T>* LinkedList<T>::getNext() const
{
	return m_next;
}

template<typename T>
void LinkedList<T>::add(LinkedList<T>* node)
{
#ifdef _DEBUG
	if (node == nullptr)
		throw std::invalid_argument("can't add nullptr.");
#endif 
	if (m_next) {
		node->add( m_next );
		m_next = node;
	}
	else {
		m_next = node;
	}
}

template<typename T>
LinkedList<T>::LinkedList() : m_value(), m_next(nullptr)
{

}

template<typename T>
LinkedList<T>::~LinkedList()
{
	std::cout << "delete: " << m_value << std::endl;
	if (m_next) {
		delete m_next;
	}
}
