#pragma once
#ifdef _WIN64

#include <windows.h>
#include <stdio.h>
#include <tchar.h>
#include <exception>

// Specify the width of the field in which to print the numbers.
// The asterisk in the format specifier "%*I64d" takes an integer
// argument and uses it to pad and right justify the number.

/// <summary>
/// Singleton class for 
/// </summary>
class MemoryCapacity
{
public:
	static MemoryCapacity& getInstance() {
		static MemoryCapacity instance;
		return instance;
	}

	enum capacity
	{
		KB = (1024ULL),
		MB = (1024ULL * 1024ULL),
		GB = (1024ULL * 1024ULL * 1024ULL)
	};

	static unsigned long long getFreeRam() {
		getMemoryStatus();
		return statex.ullAvailPhys;
	}

	static unsigned long long getFreeRam(capacity unit) {
		getMemoryStatus();
		return statex.ullAvailPhys / unit;
	}

	static unsigned long long getFreeVirtualMemory() {
		getMemoryStatus();
		return statex.ullAvailVirtual;
	}

	static unsigned long long getFreeVirtualMemory(capacity unit) {
		getMemoryStatus();
		return statex.ullAvailVirtual / unit;
	}

	/// <summary>
	/// Print MBs available in all memory
	/// </summary>
	static void print() {
		getMemoryStatus();
		const unsigned int WIDTH = 7;
		_tprintf(TEXT("There is  %*ld percent of memory in use.\n"),
			WIDTH, statex.dwMemoryLoad);
		_tprintf(TEXT("There are %*I64d total MB of physical memory.\n"),
			WIDTH, statex.ullTotalPhys / MB);
		_tprintf(TEXT("There are %*I64d free  MB of physical memory.\n"),
			WIDTH, statex.ullAvailPhys / MB);
		_tprintf(TEXT("There are %*I64d total MB of paging file.\n"),
			WIDTH, statex.ullTotalPageFile / MB);
		_tprintf(TEXT("There are %*I64d free  MB of paging file.\n"),
			WIDTH, statex.ullAvailPageFile / MB);
		_tprintf(TEXT("There are %*I64d total MB of virtual memory.\n"),
			WIDTH, statex.ullTotalVirtual / MB);
		_tprintf(TEXT("There are %*I64d free  MB of virtual memory.\n"),
			WIDTH, statex.ullAvailVirtual / MB);

		// Show the amount of extended memory available.

		_tprintf(TEXT("There are %*I64d free  MB of extended memory.\n"),
			WIDTH, statex.ullAvailExtendedVirtual / MB);
	}

	MemoryCapacity(const MemoryCapacity&) = delete;
	MemoryCapacity(MemoryCapacity&&) = delete;
	~MemoryCapacity() = default;
private:
	static MEMORYSTATUSEX statex;
	MemoryCapacity() = default;

	static void getMemoryStatus() {
		statex.dwLength = sizeof(statex); //Need to set this every time apparently
		BOOL success = GlobalMemoryStatusEx(&statex);
#ifdef _DEBUG
		if (!success)
			throw std::exception("Something is wrong with the OS or memory." + GetLastError());
#endif // _DEBUG
	}
};
MEMORYSTATUSEX MemoryCapacity::statex; //Initialize static

#else //GNU based system

class MemoryCapacity
{
public:
	static MemoryCapacity& getInstance() {
		static MemoryCapacity instance;
		return instance;
	}

	enum capacity
	{
		KB = (1024ULL),
		MB = (1024ULL * 1024ULL),
		GB = (1024ULL * 1024ULL * 1024ULL)
	};

	static unsigned long long getFreeRam() {
		throw std::exception("Not implemented.");
		return 0;
	}

	static unsigned long long getFreeRam(capacity unit) {
		throw std::exception("Not implemented.");
		return 0;
	}

	static unsigned long long getFreeVirtualMemory() {
		throw std::exception("Not implemented.");
		return 0;
	}

	static unsigned long long getFreeVirtualMemory(capacity unit) {
		throw std::exception("Not implemented.");
		return 0;
	}

	static void print() { throw std::exception("Not implemented."); }
private:

};


#endif // _WIN64 or gnu
