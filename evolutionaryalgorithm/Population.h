#pragma once
#include <vector>
#include <memory>
#include <algorithm>

#include "Individual.h"

template<typename T>
class Population 
{
public:
	Population() = default;
	~Population() = default;
	std::vector<std::unique_ptr< Individual<T> >> members;

protected:

private:
	Population(const Population& copy_this) = delete;
	Population& operator=(const Population& copy_this) = delete;

	Population(Population&& move_this) = delete;
	Population& operator=(Population&& move_this) = delete;
};
