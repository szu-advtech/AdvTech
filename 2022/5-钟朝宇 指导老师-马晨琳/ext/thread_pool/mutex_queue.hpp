#pragma once
#include <memory>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <string>
#include <vector>
#include <iostream>
#include <atomic>
#include <exception>


/**
 *  @brief 线程安全队列
 *  数据按照先进先出的顺序入队和出队,数据通过互斥锁进行保护,避免数据竞争
 */
template<typename T>
class MutexQueue {
public:

    MutexQueue(int sz = 16);
    /**
     * @brief 数据入队
     * @param t 数据
     */
    bool push(T&& t);

    bool push(const T& t);

    /**
     * @brief 尝试进行数据出队
     * @param data 从外部传入指针用于接收数据
     * @return 判断是否有数据出队
     *  @retval true 队列非空
     *  @retval false 队列为空
     */
    bool try_pop(T& data);

    /**
     * @brief 数据出队
     * @return 出队数据的拷贝
     */
    bool pop(T& data);

    /**
    * @brief 判断队列是否为空
    * @return true or false
    */
    bool empty();

    /**
     * @brief 获取队列大小
     * @return 队列的大小
     */
    size_t size() const;

    void stop();

private:
    MutexQueue(const MutexQueue &rhs) = delete;
    MutexQueue& operator=(const MutexQueue &rhs) = delete;
    std::queue<T> queue_;
    mutable std::mutex mutex;
    mutable std::condition_variable cond_nonempty;
    bool flag_stop;
};

template<typename T>
MutexQueue<T>::MutexQueue(int sz):flag_stop(false) {
}

template<typename T>
void MutexQueue<T>::stop() {
    flag_stop = true;
    cond_nonempty.notify_all();
}

template<typename T>
bool MutexQueue<T>::push(T&& t) {
    std::lock_guard<std::mutex> lock(mutex);
    queue_.push(std::move(t));
    cond_nonempty.notify_one();
    return true;
}

template<typename T>
bool MutexQueue<T>::push(const T& t) {
    std::lock_guard<std::mutex> lock(mutex);
    queue_.push(t);
    cond_nonempty.notify_one();
    return true;
}

template<typename T>
bool MutexQueue<T>::try_pop(T& data) {
    std::lock_guard<std::mutex> lock(mutex);

    if (queue_.empty()) {
        return false;
    }

    data = std::move(queue_.front());
    queue_.pop_back();
    return true;
}

template<typename T>
bool MutexQueue<T>::pop(T& data) {
    std::unique_lock<std::mutex> lock(mutex);
    cond_nonempty.wait(lock, [this]{return !queue_.empty() || flag_stop;});
    if(queue_.empty() && flag_stop) {
        return false;
    }
    data = std::move(queue_.front());
    queue_.pop();
    return true;
}

template<typename T>
size_t MutexQueue<T>::size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return queue_.size();
}

template<typename T>
bool MutexQueue<T>::empty() {
    std::lock_guard<std::mutex> lock(mutex);
    return queue_.empty();
}

