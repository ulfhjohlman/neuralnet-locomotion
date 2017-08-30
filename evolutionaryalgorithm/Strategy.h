#pragma once
#include <iostream>

class Strategy {
	virtual ~Strategy() = default;
	virtual void execute() = 0;
};

template<typename T>
class base_strategy 
{
public:
	void execute() {
		static_cast<T*>(this)->execute();
	}
};

/// <summary>
/// Example derived static strategy.
/// </summary>
class static_strategy : public base_strategy<static_strategy> {
public:
	void execute() {
		std::cout << "derived" << a << std::endl;
	}
	void set(int arg) { a = arg; }
private:
	int a;
};