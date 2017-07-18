#pragma once

#include <thread>
#include <vector>
#include <mutex>
#include <queue>
#include <functional>
#include <condition_variable>
#include <iostream>
#include <atomic>

#include "../utility/FunctorWrapper.h"

/// <summary>
/// Summary goes here
/// </summary>
class SimpleThreadPool
{
public:
	SimpleThreadPool() : m_done(false) {
		m_number_of_threads = std::thread::hardware_concurrency() - 1;
#pragma warning(disable:4267)
		m_number_of_working_threads.store(m_number_of_threads);
#pragma warning(default:4267)

		for (size_t i = 0; i < m_number_of_threads; i++) {
			m_workers.push_back(std::thread(&SimpleThreadPool::worker, this));
		}
	}
	~SimpleThreadPool() {

		//Stop all waiting threads. 
		m_done.store(true);
		m_cond.notify_all();
		
		//Join all threads
		unsigned int k = 0;
		for (auto& i : m_workers) 
			if (i.joinable()) {
				i.join();
				k++;
			}

		std::cout << "SimpleThreadPool closing " << k << " threads." << std::endl;
	}

	void addWork(std::function<void()> f) {
		std::lock_guard<std::mutex> lk(m_mutex);
		m_queue.push(f);
		m_cond.notify_one();
	}

	/// <summary>
	/// returns true if all work is done and queue is empty.
	/// </summary>
	bool isDone() {
		std::lock_guard<std::mutex> lk(m_mutex);
		return this->m_queue.empty() && m_number_of_working_threads.load() == 0;
	}

	/// <summary>
	/// returns true if work fetched from queue, false otherwise.
	/// </summary>
	/// <param name="f"></param>
	/// <returns></returns>
	bool tryPopWork(std::function<void()>& f) {
		std::lock_guard<std::mutex> lk(m_mutex);
		if (m_queue.empty())
			return false;
		f = m_queue.front();
		m_queue.pop();
		return true;
	}

	/// <summary>
	/// Helps pool with remaining work in queue. Returns
	/// when queue is empty.
	/// </summary>
	void help() 
	{
		std::function<void()> f = [] {};
		while (tryPopWork(f)) 
			f();
	}

protected:
	/// <summary>
	/// Wait for a maximum of 500ms for work to appear in queue before timeout.
	/// returns dummy function in later case.
	/// </summary>
	/// <param name="f"></param>
	void waitForWork(std::function<void()>& f) {
		f = [] {}; //set dummy function
		m_number_of_working_threads.fetch_sub(1);

		std::unique_lock<std::mutex> lk(m_mutex);
		m_cond.wait(lk, [this] { return !this->m_queue.empty() || m_done.load(); });
		if (m_done) //done and return dummy
			return;

		m_number_of_working_threads.fetch_add(1);
		f = m_queue.front();
		m_queue.pop();
	}

	/// <summary>
	/// Thread pool worker. Finishes when either exception is thrown or 
	/// object is destructed and all work is done.
	/// </summary>
	void worker() {
		using namespace std::chrono_literals;
		while (!m_done.load()) {
			std::function<void()> task;
			waitForWork(task);
			task();
		}
	}

	size_t m_number_of_threads;
	std::vector<std::thread> m_workers;
	std::queue<std::function<void()>> m_queue;

	std::mutex m_mutex;
	std::condition_variable m_cond;

	std::atomic_bool m_done;
	std::atomic_char m_number_of_working_threads;
};
