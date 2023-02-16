#pragma once

#include <cstdint>
#include <utility>

template<typename Handler>
struct TaskTracker{
    TaskTracker() {}
    TaskTracker(Handler&& handler): handler(std::forward<Handler>(handler)){
    }

    Handler handler;
};
