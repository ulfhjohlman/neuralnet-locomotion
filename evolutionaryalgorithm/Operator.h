#pragma once




class base_operator {
	virtual ~base_operator() = default;
};

template<typename T>
class sub_operator : public base_operator {
	
};
