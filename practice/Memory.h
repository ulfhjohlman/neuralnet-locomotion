#pragma once
#include "MemoryCapacity.h"
#include <memory>

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
	/// Will always deny if there is less then 200 MB ram left.
	/// or set limit. Will not consider virtual memory yet.
	/// </summary>
	template<typename T>
	static auto requestAllocation(size_t size, size_t minimum_bytes_left = MemoryCapacity::MB * 200ULL) {
		size_t memory_required = size * sizeof(T);
		size_t memory_available = MemoryCapacity::getFreeRam();
		if (memory_available - memory_required < minimum_bytes_left)
			return (T*)nullptr;
		return new T[size];
	}

	Memory(const Memory&) = delete;
	Memory(Memory&&) = delete;
	~Memory() = default;
private:
	Memory() = default;
};
