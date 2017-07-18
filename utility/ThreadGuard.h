#pragma once
#include <thread>
class ThreadGuard {
public:
	explicit ThreadGuard(std::thread& t_) : t(t_) { }
	~ThreadGuard() {
		if (t.joinable()) {
			t.join();
		}
	}
	ThreadGuard(ThreadGuard const&) = delete;
	ThreadGuard& operator=(ThreadGuard const&) = delete;
private:
	std::thread& t;
};