#pragma once

#include <memory>

/// <summary>
/// A concrete class able to call the explicit functor
/// from a family of functors.
/// </summary>
class FunctorWrapper
{
public:	
	FunctorWrapper() = default;

	template<typename F>
	FunctorWrapper(F&& f) : //wraps the functor to the type family.
		implementation(new type<F>(std::move(f))) {} //move ownership

	FunctorWrapper(FunctorWrapper&& move_this) : //move "move_this" implementation
		implementation(std::move(move_this.implementation)) { }

	FunctorWrapper& operator=(FunctorWrapper&& move_this) { //same as 1 above
		implementation = std::move(move_this.implementation);
		return *this;
	}

	/// <summary>
	/// Call implementation.
	/// </summary>
	void operator()() { implementation->call(); }

	FunctorWrapper(const FunctorWrapper&) = delete;
	FunctorWrapper& operator=(const FunctorWrapper&) = delete;
private:

	/// <summary>
	/// A base class for template child classes.
	/// </summary>
	struct type_family
	{
		virtual ~type_family() = default;
		virtual void call() = 0;
	};

	template<typename F>
	struct type : type_family
	{
		F m_f;
		type(F&& f) : m_f( std::move(f) ) {} // move constructor
		void call() { m_f(); }
	};

	std::unique_ptr<type_family> implementation;
};

