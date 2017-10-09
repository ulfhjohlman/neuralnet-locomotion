#pragma once
#include <queue>
#include <mutex>
#include <memory>

#include "ThreadsafeQueue.h"

template< typename Agent >
class AgentScheduler
{
public:
	AgentScheduler( int number_of_parallel_simulations = 20 ) : 
		m_number_of_parallel_simulations( number_of_parallel_simulations ) { }
	~AgentScheduler() = default;

	std::unique_ptr< Agent > next() {
		std::unique_lock<std::mutex> lk_active(m_exclude_active);
		std::unique_lock<std::mutex> lk_inactive(m_exclude_inactive, std::defer_lock);

		if (!m_active_agents.empty()) {
			auto holder = std::move(m_active_agents.front());
			m_active_agents.pop();
			return holder;
		}
		else {
			lk_active.unlock();
			lk_inactive.lock();
			if (!m_inactive_agents.empty()) {
				auto holder = std::move(m_inactive_agents.front());
				m_inactive_agents.pop();
				return holder;
			}
		}
		return nullptr; //nothing to get from schedule
	}

	//Add one from inactive to active slot in schedule
	void fill() {
		std::lock(m_exclude_active, m_exclude_inactive);
		std::lock_guard<std::mutex> lk1(m_exclude_active, std::adopt_lock);
		std::lock_guard<std::mutex> lk2(m_exclude_inactive, std::adopt_lock);

		if (!m_inactive_agents.empty()) {
			auto agent = std::move(m_inactive_agents.front());
			m_inactive_agents.pop();
			m_active_agents.push(std::move(agent));
		}
	}

	void returnToSchedule( std::unique_ptr< Agent > pAgent ) {
		std::lock_guard<std::mutex> lk_active(m_exclude_active);
		m_active_agents.push(std::move(pAgent));
	}
	
	void schedule(std::unique_ptr< Agent > pAgent) {
		std::unique_lock<std::mutex> lk_active(m_exclude_active);
		std::unique_lock<std::mutex> lk_inactive(m_exclude_inactive, std::defer_lock);

		if (m_active_agents.size() < m_number_of_parallel_simulations) {
			m_active_agents.push(std::move(pAgent)); //max out parallel simulations host have access to.
		}
		else {
			lk_active.unlock();
			lk_inactive.lock();
			m_inactive_agents.push(std::move(pAgent));//else put em on a list.
		}
	}
	size_t size() {
		std::lock_guard<std::mutex> lk(m_exclude_active);
		return m_active_agents.size();
	}
	
private:
	std::queue< std::unique_ptr<Agent> > m_active_agents;
	std::queue< std::unique_ptr<Agent> > m_inactive_agents;
	std::mutex m_exclude_active;
	std::mutex m_exclude_inactive;

	int m_number_of_parallel_simulations;

public:
	AgentScheduler(const AgentScheduler& copy_this) = delete;
	AgentScheduler& operator=(const AgentScheduler& copy_this) = delete;

	AgentScheduler(AgentScheduler&& move_this) = delete;
	AgentScheduler& operator=(AgentScheduler&& move_this) = delete;
};