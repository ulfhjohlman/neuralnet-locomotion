#pragma once
#include <sstream>
#include <vector>
#include <exception>
#include <type_traits>
#include <stdexcept>

/// <summary>
/// Just a testing class
/// </summary>
class DataPrinter
{
public:
	DataPrinter() {}
	virtual ~DataPrinter() = default;

	template<typename T>
	void write(std::vector<T>& data) {
		static_assert(std::is_integral<T>::value || std::is_arithmetic<T>::value, "Not a number. Can only store number data.");
		//m_buffer.str().reserve( data.size() * sizeof(data) * 4 );
			for (const auto& i : data) {
				m_buffer << i << " ";
				if (!m_buffer)
					throw std::runtime_error("Unable to write to buffer in data printer.");
			}
	}


	std::string getString() const {
		return m_buffer.str();
	}
private:
	std::ostringstream m_buffer;
};

