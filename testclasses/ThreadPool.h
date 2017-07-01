#pragma once
#include"ThreadsafeQueue.h"
#include"FunctionWrapper.h"

#include<thread>
#include<future>
#include<atomic>
#include<functional>
#include<vector>
#include<exception>
#include<memory>

class ThreadPool
{
public:
	ThreadPool();
	~ThreadPool();

	template<typename Function>
	std::future<typename std::result_of< Function() >::type > submit(Function f);

private:
	void workerThread();

	std::atomic_bool m_done;
	ThreadsafeQueue< FunctionWrapper > m_queue;
	std::vector<std::thread> m_threads;
};

ThreadPool::ThreadPool() : m_done(false)
{
	const size_t thread_count = std::thread::hardware_concurrency();
	try {
		for (size_t i = 0; i < thread_count; i++) {
			m_threads.push_back( std::thread( &ThreadPool::workerThread, this ));
		} 
	} catch (...) {
		m_done = true;
		throw;
	}

}

ThreadPool::~ThreadPool()
{
	m_done = true;
	for (size_t i = 0; i<m_threads.size(); ++i) {
		if (m_threads[i].joinable())
			m_threads[i].join();
	}
}

inline void ThreadPool::workerThread() {
	while (!m_done) {
		FunctionWrapper task;
		if (m_queue.try_pop(task)) {
			task();
		}
		else {
			std::this_thread::yield();
		}

	}
}

template<typename Function>
inline std::future<typename std::result_of<Function()>::type> ThreadPool::submit(Function f)
{
	typedef typename std::result_of<Function()>::type result_type;

	std::packaged_task< result_type() > task(std::move(f));
	std::future<result_type> result(task.get_future());
	m_queue.push(std::move(task));

	return result;
}
