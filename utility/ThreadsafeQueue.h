#pragma once
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>


/// <summary>
/// A thread safe queue implementation. Has move assignment construction.
/// Pretty much complete, lacking emplace. Should implement move/emplace range.
/// </summary>
template<typename T>
class ThreadsafeQueue
{
public:
	ThreadsafeQueue() = default;
	~ThreadsafeQueue() = default;
	ThreadsafeQueue& operator=(ThreadsafeQueue&& move_this);
	
	//Moves the copy into queue. Pass by std::move if big object, or unique_ptr
	void push(T new_value);

	//Return true if value could be retrieved. False otherwise
	bool try_pop(T& value);

	//Blocks until a value can be retrieved from queue.
	void wait_and_pop(T& value);

	//Blocks until a value can be assigned from queue.
	T wait_and_pop();

	//Not thread safe pop get and pop.
	T sequential_pop();

	bool empty() const;
	size_t size() const;
private:
	mutable std::mutex m_mut;
	std::queue<T> m_queue;
	std::condition_variable m_condition;
public:
	ThreadsafeQueue(const ThreadsafeQueue&) = delete;
	ThreadsafeQueue& operator=(const ThreadsafeQueue&) = delete;
	ThreadsafeQueue(ThreadsafeQueue&& move_this) = delete;
};

template<typename T>
ThreadsafeQueue<T>& ThreadsafeQueue<T>::operator=(ThreadsafeQueue<T>&& move_this)
{
	while (true) {
		T holder;
		bool can_pop = move_this.try_pop(T);
		if (can_pop)
			m_queue.push(holder);
		else
			break;
	}
	return *this;
}

template<typename T>
void ThreadsafeQueue<T>::push(T new_value)
{
	std::lock_guard<std::mutex> lk(m_mut);
	m_queue.push(std::move(new_value));
	m_condition.notify_one();
}

template<typename T>
void ThreadsafeQueue<T>::wait_and_pop(T & value)
{
	std::unique_lock<std::mutex> lk(m_mut);
	m_condition.wait(lk, [this] { return !m_queue.empty(); });
	value = std::move(m_queue.front());
	m_queue.pop();
}

template<typename T>
T ThreadsafeQueue<T>::wait_and_pop() {
	std::unique_lock<std::mutex> lk(m_mut);
	m_condition.wait(lk, [this] { return !m_queue.empty(); });
	auto holder = std::move(m_queue.front());
	m_queue.pop();
	return holder;
}

template<typename T>
T ThreadsafeQueue<T>::sequential_pop()
{
#ifdef _DEBUG
	if (m_queue.empty())
		throw std::runtime_error("Empty queue tried to pop.");
#endif // _DEBUG
	auto holder = std::move(m_queue.front());
	m_queue.pop();
	return holder;
}

template<typename T>
bool ThreadsafeQueue<T>::try_pop(T& value)
{
	std::lock_guard<std::mutex> lk(m_mut);
	if (m_queue.empty())
		return false;
	value = std::move(m_queue.front());
	m_queue.pop();
	return true;
}

template<typename T>
bool ThreadsafeQueue<T>::empty() const
{
	std::lock_guard<std::mutex> lk(m_mut);
	return m_queue.empty();
}

template<typename T>
size_t ThreadsafeQueue<T>::size() const
{
	std::lock_guard<std::mutex> lk(m_mut);
	return m_queue.size();
}

