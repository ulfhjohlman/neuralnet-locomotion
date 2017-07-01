#include <iostream>
#include <vector>
#include <memory>
#include <exception>
#include <type_traits>
#include<thread>
#include <chrono>

#include <windows.h>
#include <stdio.h>
#include <tchar.h>

// Use to convert bytes to MB
#define DIV (1024*1024)

// Specify the width of the field in which to print the numbers.
// The asterisk in the format specifier "%*I64d" takes an integer
// argument and uses it to pad and right justify the number.
#define WIDTH 7

void memory_test();

int main() {
	MEMORYSTATUSEX statex;

	statex.dwLength = sizeof(statex);

	GlobalMemoryStatusEx(&statex);

	_tprintf(TEXT("There is  %*ld percent of memory in use.\n"),
		WIDTH, statex.dwMemoryLoad);
	_tprintf(TEXT("There are %*I64d total MB of physical memory.\n"),
		WIDTH, statex.ullTotalPhys / DIV);
	_tprintf(TEXT("There are %*I64d free  MB of physical memory.\n"),
		WIDTH, statex.ullAvailPhys / DIV);
	_tprintf(TEXT("There are %*I64d total MB of paging file.\n"),
		WIDTH, statex.ullTotalPageFile / DIV);
	_tprintf(TEXT("There are %*I64d free  MB of paging file.\n"),
		WIDTH, statex.ullAvailPageFile / DIV);
	_tprintf(TEXT("There are %*I64d total MB of virtual memory.\n"),
		WIDTH, statex.ullTotalVirtual / DIV);
	_tprintf(TEXT("There are %*I64d free  MB of virtual memory.\n"),
		WIDTH, statex.ullAvailVirtual / DIV);

	// Show the amount of extended memory available.

	_tprintf(TEXT("There are %*I64d free  MB of extended memory.\n"),
		WIDTH, statex.ullAvailExtendedVirtual / DIV);

	//memory_test(); // don't uncomment this
}

void memory_test()
{
	using namespace std::chrono_literals;
	std::vector<std::unique_ptr< char >> chunks;

	auto allocate_chunk = [](size_t size) { return new char[size]; };

	try {
		while (true) {
			std::this_thread::sleep_for(1000ms);
			chunks.push_back(std::unique_ptr<char>(allocate_chunk(1024 * 1024 * 1024))); //allocate 1GB chunks
		}
	}
	catch (std::bad_alloc& e) {
		std::cerr << e.what() << std::endl;
	}
}