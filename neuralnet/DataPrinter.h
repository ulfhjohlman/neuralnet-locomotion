#pragma once
#include <sstream>
#include <vector>
#include <exception>

class DataPrinter
{
public:
	DataPrinter() = default;
	virtual ~DataPrinter() = default;

	template<typename T>
	void write(T data) {
		m_buffer << m_buffer.scientific << data;
	}

	template<typename T>
	void write(std::vector<T>& data) {
			for (const auto& i : data) {
				m_buffer << i << " ";
			}
	}

	std::string getString() const {
		return m_buffer.str();
	}
	const char* getData() const {
#ifdef _DEBUG
		throw std::exception("Fuck this function, try printing");
#endif // _DEBUG

		return m_buffer.str().c_str();
	}


private:
	std::ostringstream m_buffer;
};

