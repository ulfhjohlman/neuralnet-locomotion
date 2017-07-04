#pragma once
#include "MemoryCapacity.h"
#include <memory>
#include <exception>
#include <type_traits>

class Memory
{
public:
	static Memory& getInstance() {
		static Memory instance;
		return instance;
	}

	/// <summary>
	/// Request allocation of type T of said size. Will return
	/// nullptr if request is denied./r/n
	///
	/// Can throw bad_alloc.
	///
	/// Will always denie if there is less then 200 MB ram left.
	/// or set limit. Will not consider virtual memory yet.
	/// </summary>
	template<typename T>
	static auto requestAllocation(size_t size) {
		unsigned long long memory_required = size * sizeof(T);
		unsigned long long memory_available = MemoryCapacity::getFreeRam();
		if (memory_available - memory_required < MemoryCapacity::MB * 200ULL)
			return (T*)nullptr;
		return new T[size];
	}

	Memory(const Memory&) = delete;
	Memory(Memory&&) = delete;
	~Memory() = default;
private:
	Memory() = default;
};
