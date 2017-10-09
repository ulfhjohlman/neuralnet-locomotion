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
/// Todo: change to thread safe queue container
/// </summary>
class ThreadPool
{
public:
	ThreadPool() : m_done(false) {
		//Leave one thread for host to avoid over subscription
		m_number_of_threads = std::thread::hardware_concurrency() - 1u;
		m_number_of_working_threads.store(m_number_of_threads);

		//Launch workers
		for (size_t i = 0; i < m_number_of_threads; i++) {
			m_workers.push_back(std::thread(&ThreadPool::worker, this));
		}
	}
	~ThreadPool() {
		//Host thread is trying to destruct
		//Finish up all work left.
		this->finish();

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
#ifdef _DEBUG
		if (k < m_number_of_threads) {
			std::cerr << "ThreadPool closing " << k << " threads." << std::endl;
			std::cerr << "Some threads died along the way, very critical error. Press any key to continue..." << std::endl;
			std::cin.get();
		}
#endif // _DEBUG
	}

	/// <summary>
	/// Add work item with no explicit return value or synchronization option.
	/// </summary>
	/// <param name="f">Is a right hand side reference argument, use std::move to avoid unmoved structures.</param>
	void addWork(FunctorWrapper&& f) {
		std::unique_lock<std::mutex> lk(m_mutex);
		m_queue.push(std::move(f));

		lk.unlock();
		m_cond.notify_one();
	}

	/// <summary>
	/// Returns a future for the return type 
	/// of functor f.
	/// <para/>Ex: 
	/// <para/>auto fut = ::submit( []() -> int { return 1; } ) 
	/// <para/>fut.get(); //returns (int)1 
	/// </summary>
	template<typename Functor>
	std::future<typename std::result_of<Functor()>::type> submit(Functor f)
	{
		typedef typename std::result_of<Functor()>::type result_type;

		// packaged_task packs away f and then pushed to queue for processing
		std::packaged_task< result_type() > task(std::move(f));
		std::future<result_type> result(task.get_future());

		std::unique_lock<std::mutex> lk(m_mutex);
		m_queue.push(std::move(task));

		lk.unlock();
		m_cond.notify_one();

		//Result from packaged_task task will be stored in future "result"
		return result; 
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
	bool tryPopWork(FunctorWrapper& f) {
		std::lock_guard<std::mutex> lk(m_mutex);
		if (m_queue.empty())
			return false;
		f = std::move(m_queue.front());
		m_queue.pop();
		return true;
	}

	/// <summary>
	/// Helps pool with remaining work in queue. Returns
	/// when queue is empty, but not necessarily done.
	/// </summary>
	void help()
	{
		FunctorWrapper f = []{};
		while (tryPopWork(f))
			f();
	}

	/// <summary>
	/// Not tested, but should returns when all work items are finished.
	/// Can be used by host thread to synchronize all operations.
	/// </summary>
	void finish() {
		FunctorWrapper f = [] {};
		while (m_done)
		{
			bool can_work = tryPopWork(f);
			if (can_work)
				f();
			else
				std::this_thread::yield();
		}
	}

protected:
	/// <summary>
	/// Waiting for work in sleep mode. Does not consume processor time if no work. 
	/// </summary>
	/// <param name="f"></param>
	void waitForWork(FunctorWrapper& f) {
		f = [] {}; //set dummy function
		m_number_of_working_threads.fetch_sub(1u); //This thread is not working

		std::unique_lock<std::mutex> lk(m_mutex);
		m_cond.wait(lk, [this] { return !m_queue.empty() || m_done.load(); });
		if (m_done) //done and return dummy
			return;

		m_number_of_working_threads.fetch_add(1u); //This thread is working
		f = std::move(m_queue.front());
		m_queue.pop();

		lk.unlock();
		m_cond.notify_one(); 
	}

	/// <summary>
	/// Thread pool worker. Finishes when either exception is thrown or 
	/// object is destructed and all work is done.
	/// </summary>
	void worker() {
		using namespace std::chrono_literals;
		while (!m_done.load()) {
			FunctorWrapper task;
			waitForWork(task);
			task();
		}
	}

	unsigned int m_number_of_threads;
	std::vector<std::thread> m_workers;
	std::queue<FunctorWrapper> m_queue;

	std::mutex m_mutex;
	std::condition_variable m_cond;

	std::atomic_bool m_done;
	std::atomic_uint m_number_of_working_threads;
};
