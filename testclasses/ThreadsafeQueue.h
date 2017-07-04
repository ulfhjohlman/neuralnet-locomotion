#pragma once
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>

template<typename T>
class ThreadsafeQueue
{
public:
	ThreadsafeQueue() = default;
	ThreadsafeQueue(const ThreadsafeQueue&) = default;
	ThreadsafeQueue& operator=(
		const ThreadsafeQueue&) = delete;

	void push(T new_value);
	bool try_pop(T& value);
	std::shared_ptr<T> try_pop();
	void wait_and_pop(T& value);
	std::shared_ptr<T> wait_and_pop();
	bool empty() const;

private:
	mutable std::mutex m_mut;
	std::queue<T> m_queue;
	std::condition_variable m_condition;
};

template<typename T>
void ThreadsafeQueue<T>::push(T new_value)
{
	std::lock_guard<std::mutex> lk(m_mut);
	m_queue.push(new_value);
	m_condition.notify_one();
}

template<typename T>
void ThreadsafeQueue<T>::wait_and_pop(T & value)
{
	std::unique_lock<std::mutex> lk(m_mut);
	m_condition.wait(lk, [this] { return !m_queue.empty(); });
	value = m_queue.front();
	m_queue.pop();
}

template<typename T>
std::shared_ptr<T> ThreadsafeQueue<T>::wait_and_pop()
{
	std::unique_lock<std::mutex> lk(m_mut);
	m_condition(lk, [this] { return !m_queue.empty(); });
	std::shared_ptr<T> sp( std::make_shared<T>(m_queue.front()) );
	m_queue.pop();
	return sp;
}

template<typename T>
bool ThreadsafeQueue<T>::try_pop(T& value)
{
	std::lock_guard<std::mutex> lk(m_mut);
	if (m_queue.empty())
		return false;
	T new_value( std::move( m_queue.front() ));
	//value = std::move(m_queue.front()) ;
	m_queue.pop();
	return true;
}

template<typename T>
std::shared_ptr<T> ThreadsafeQueue<T>::try_pop()
{
	std::lock_guard<std::mutex> lk(m_mut);
	std::shared_ptr<T> sp = nullptr;
	if (m_queue.empty())
		return sp;
	sp = std::make_shared<T>(m_queue.front());
	m_queue.pop();
	return sp;
}

template<typename T>
bool ThreadsafeQueue<T>::empty() const
{
	std::lock_guard<std::mutex> lk(m_mut);
	return m_queue.empty(); //Can be inconsistent while being modified by another thread
}

