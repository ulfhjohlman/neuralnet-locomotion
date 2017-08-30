#include "stdafx.h"

#include <bitset>
#include <iostream>

#ifdef _DEBUG
#pragma comment(lib, "../x64/Debug/evolutionaryalgorithm.lib")
#else
#pragma comment(lib, "../x64/Release/evolutionaryalgorithm.lib")
#endif // _DEBUG

	

int main()
{
	std::cout << ((5 ^ 3)^3) << std::endl;
	return 0;
}