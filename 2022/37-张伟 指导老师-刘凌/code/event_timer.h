#ifndef PCTimer_h__
#define PCTimer_h__
 

#include <string>
#include <unordered_map>
#include <ctime>


#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && !defined(__CYGWIN__)
// if define windows, we use windows.h
#include <windows.h>

/// Performance counter timer by using Windows API. (for Windows)
template <class TKey>
class PCTimerT
{
public:
	typedef TKey key_type;
	typedef LARGE_INTEGER value_type;
	typedef std::unordered_map<key_type, value_type> tmap;
	typedef std::pair<key_type, value_type> pair_type;

protected:
	tmap events;
	value_type freq;

public:
	PCTimerT() { QueryPerformanceFrequency(&freq); }
	
	value_type GetTimeEvent(key_type name) { return events[name]; }
	bool HasTimeEvent(key_type name) { return (events.find(name) != events.end()); }

	void SetTimeEvent( key_type name )
	{
		value_type now;
		QueryPerformanceCounter(&now);
		if (HasTimeEvent(name)) {
			events[name] = now;
		} else {
			pair_type p = std::make_pair(name, now);
			events.insert(p);
		}
	}

	double TimeSpan( key_type nameFrom, key_type nameTo )
	{
		return (double)(GetTimeEvent(nameTo).QuadPart - GetTimeEvent(nameFrom).QuadPart) / freq.QuadPart;
	}
};

#else
// if windows is not defined, we use chrono from C++11
#include <chrono>

/// Performance counter timer by using C++11 chrono. (for other platforms)
template <class TKey>
class PCTimerT
{
public:
	typedef TKey key_type;
	typedef std::chrono::high_resolution_clock::time_point value_type;
	typedef std::unordered_map<key_type, value_type> tmap;
	typedef std::pair<key_type, value_type> pair_type;

protected:
	tmap events;

public:
	PCTimerT() {}
	value_type GetTimeEvent(key_type name) { return events[name]; }
	bool HasTimeEvent(key_type name) { return (events.find(name) != events.end()); }

	void SetTimeEvent(key_type name)
	{
		value_type now = std::chrono::high_resolution_clock::now();
		if (HasTimeEvent(name)) {
			events[name] = now;
		}
		else {
			pair_type p = std::make_pair(name, now);
			events.insert(p);
		}
	}

	double TimeSpan(key_type nameFrom, key_type nameTo)
	{
		std::chrono::duration<double, std::nano> d =
			GetTimeEvent(nameTo) - GetTimeEvent(nameFrom);	
		return ((double)d.count() * std::nano::num / std::nano::den);
	}
};
#endif 

/// PCTimer that uses string as its key
typedef PCTimerT<std::string> EventTimer;


#endif // PCTimer_h__
