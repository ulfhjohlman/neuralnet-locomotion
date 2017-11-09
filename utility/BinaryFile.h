#pragma once
#include <fstream>
#include <iomanip>
#include <string>
#include <stdexcept>

class BinaryFile 
{
public:
	BinaryFile(const char* file_name) : m_file_name(file_name) { }
	BinaryFile(const std::string& file_name) : m_file_name(file_name) { }
	~BinaryFile() = default;

	BinaryFile& operator=(const char* file_name) { m_file_name = file_name; return *this; }
	BinaryFile& operator=(const std::string& file_name) { m_file_name = file_name; return *this; }

	void setFileName(const char* file_name) { m_file_name = file_name; }
	void setFileName(const std::string& file_name) { m_file_name = file_name; }

	const char* fileName() const { return m_file_name.c_str(); }
	
	template<typename T, typename size_type = std::streamsize>
	void write(const T* data, size_type size) {
		std::fstream binary_output;
		binary_output.open(m_file_name, std::ios::out | std::ios::binary | std::ios::trunc);
		if (binary_output.is_open()) {
			//write number of bytes in data type
			binary_output.seekp(0);
			char n_bytes = sizeof(T);
			binary_output.write((char*)&n_bytes, sizeof(n_bytes));

			//write size
			binary_output.write((char*)&size, sizeof(size));

			//write data
			binary_output.write((char*)data, size * sizeof(T));
			binary_output.close();
		}
		else {
			throw std::runtime_error("Cannot open binary file output stream.");
		}
	}
	template<typename T, typename size_type = std::streamsize>
	void read(T* data, size_type size) {
		std::fstream binary_input;
		binary_input.open(m_file_name, std::ios::in | std::ios::binary);
		if (binary_input.is_open()) {
			//read number of bytes in data type
			char n_bytes;
			binary_input.read((char*)&n_bytes, sizeof(n_bytes));

			if (n_bytes != sizeof(T))
				throw std::runtime_error("Very bad data type mismatch in binary file read.");

			// read the number of elements 
			size_type size_in_file;
			binary_input.read((char*)&size_in_file, sizeof(size_in_file)); 

			if (size_in_file != size)
				throw std::runtime_error("Mismatch number of elements in array and file.");

			// read data
			binary_input.read((char*)data, size * sizeof(T));
			binary_input.close();
		}
		else {
			throw std::runtime_error("Cannot open binary file input stream.");
		}
	}

private:
	std::string m_file_name;

public:
	BinaryFile(const BinaryFile& copy_this) = delete;
	BinaryFile& operator=(const BinaryFile& copy_this) = delete;

	BinaryFile(BinaryFile&& move_this) = delete;
	BinaryFile& operator=(BinaryFile&& move_this) = delete;
};