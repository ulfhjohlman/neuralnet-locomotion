#pragma once
#include <sstream>
#include <vector>
#include <exception>
#include <type_traits>

class DataPrinter
{
public:
	DataPrinter() = default;
	virtual ~DataPrinter() = default;

	template<typename T>
	void write(T data) {
		static_assert(std::is_integral<T>::value || std::is_arithmetic<T>::value, "Not a number. Can only store number data.");
		m_buffer << m_buffer.scientific << data;
	}

	template<typename T>
	void write(std::vector<T>& data) {
		static_assert(std::is_integral<T>::value || std::is_arithmetic<T>::value, "Not a number. Can only store number data.");
			for (const auto& i : data) {
				m_buffer << i << " ";
			}
	}

	const std::string& getString() const {
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

