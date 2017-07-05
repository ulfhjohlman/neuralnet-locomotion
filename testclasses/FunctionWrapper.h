#pragma once
#include <memory>
#include <type_traits>

/// <summary>
/// Function wrapper to a to void functions with no arguments. Template arguments must be 
/// movable. This is so that it can be used in std::packaged_task which can only move and not
/// copyable.
/// </summary>
class FunctionWrapper
{
public:
	FunctionWrapper() = default;

	template<typename F>
	FunctionWrapper(F&& f) :
		impl(new impl_type<F>(std::move(f))) { }
	FunctionWrapper(FunctionWrapper&& other) :
		impl(std::move(other.impl)) { }
	FunctionWrapper& operator=(FunctionWrapper&& other)
	{
		impl = std::move(other.impl);
		return *this;
	}

	//void functional operator for this object.
	void operator()() { impl->call(); } 
private:
	/// <summary>
	/// Implementation base
	/// </summary>
	struct impl_base {
		virtual void call() = 0;
		virtual ~impl_base() {}
	};

	/// <summary>
	/// Implementation type
	/// </summary>
	template<typename F>
	struct impl_type : impl_base
	{
		F f;
		impl_type(F&& f_) : f(std::move(f_)) {}
		void call() { f(); }
	};

	std::unique_ptr<impl_base> impl;

	/*FunctionWrapper(const FunctionWrapper&) = delete;
	FunctionWrapper(FunctionWrapper&) = delete;
	FunctionWrapper& operator=(FunctionWrapper& rhs) = delete;*/
};