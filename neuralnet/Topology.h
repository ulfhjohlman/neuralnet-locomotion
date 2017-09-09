#pragma once

class Topology 
{
public:
	Topology() = default;
	~Topology() = default;
	
	Topology(const Topology& copy_this) = delete;
	Topology& operator=(const Topology& copy_this) = delete;
	
	Topology(Topology&& move_this) = delete;
	Topology& operator=(Topology&& move_this) = delete;
private:
};
