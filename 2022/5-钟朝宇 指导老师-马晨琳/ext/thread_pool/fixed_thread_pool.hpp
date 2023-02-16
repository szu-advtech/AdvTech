#pragma once

#include "fixed_function.hpp"
#include "mpmc_bounded_queue.hpp"
#include "thread_pool_options.hpp"
#include "thread_worker.hpp"
#include "thread_task_tracker.hpp"
#include <atomic>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>
#include "mutex_queue.hpp"

namespace tp
{

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
class ThreadPoolImpl;
using ThreadPool = ThreadPoolImpl<FixedFunction<void(), 256>, TaskTracker,
                                  MPMCBoundedQueue>;
using IOThreadPool = ThreadPoolImpl<FixedFunction<void(), 256>, TaskTracker,
        MutexQueue>;
/**
 * @brief The ThreadPool class implements thread pool pattern.
 * It is highly scalable and fast.
 * It is header only.
 * It implements both work-stealing and work-distribution balancing
 * startegies.
 * It implements cooperative scheduling strategy for tasks.
 */
template<typename Task, template<typename> class Tracker, template<typename> class Queue>
class ThreadPoolImpl {
public:
    /**
     * @brief ThreadPool Construct and start new thread pool.
     * @param options Creation options.
     */
    explicit ThreadPoolImpl(
        const ThreadPoolOptions& options = ThreadPoolOptions(0));

    /**
     * @brief Move ctor implementation.
     * @param rhs ThreadPoolImpl对象右值
     */
    ThreadPoolImpl(ThreadPoolImpl&& rhs) noexcept;

    /**
     * @brief ~ThreadPool Stop all workers and destroy thread pool.
     */
    ~ThreadPoolImpl();

    /**
     * @brief Move assignment implementaion.
     * @param rhs ThreadPoolImpl对象右值
     */
    ThreadPoolImpl& operator=(ThreadPoolImpl&& rhs) noexcept;

    /**
         * @brief post Try post job to thread pool.
         * @param handler Handler to be called from thread pool worker. It has
         * to be callable as 'handler()'.
         * @return 'true' on success, false otherwise.
         * @note All exceptions thrown by handler will be suppressed.
         */
    template<typename Handler>
    bool tryPost(Handler &&handler);

    /**
     * @brief post Post job to thread pool.
     * @param handler Handler to be called from thread pool worker. It has
     * to be callable as 'handler()'.
     * @throw std::overflow_error if worker's queue is full.
     * @note All exceptions thrown by handler will be suppressed.
     */
    template<typename Handler>
    void post(Handler &&handler);

    void stop() {
        for (auto& worker_ptr : m_workers)
        {
            worker_ptr->stop();
        }
    }


private:
    Worker<Task, Tracker, Queue>& getWorker();

    std::vector<std::unique_ptr<Worker<Task, Tracker, Queue>>> m_workers;
    tp::ThreadPoolImpl<Task, Tracker, Queue> *thread_pool;
    std::atomic<size_t> m_next_worker;
    std::atomic<uint64_t> workload;

};


/// Implementation

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
inline ThreadPoolImpl<Task, Tracker, Queue>::ThreadPoolImpl(const ThreadPoolOptions& options)
    : m_workers(options.threadCount())
    , m_next_worker(0)
    , workload(0)
{
    for(auto& worker_ptr : m_workers) {
        worker_ptr.reset(new Worker<Task, Tracker, Queue>(options.queueSize(), this));
    }

    for(size_t i = 0; i < m_workers.size(); ++i) {
        if(options.isIOPreference()) {
            m_workers[i]->start(i);
        } else {
            Worker<Task, Tracker, Queue>* steal_donor =
                    m_workers[(i + 1) % m_workers.size()].get();
            m_workers[i]->start(i, steal_donor);
        }
    }
}

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
inline ThreadPoolImpl<Task, Tracker, Queue>::ThreadPoolImpl(ThreadPoolImpl<Task, Tracker, Queue>&& rhs) noexcept
{
    *this = rhs;
}

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
inline ThreadPoolImpl<Task, Tracker, Queue>::~ThreadPoolImpl()
{
    stop();
}

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
inline ThreadPoolImpl<Task, Tracker, Queue>&
ThreadPoolImpl<Task, Tracker, Queue>::operator=(ThreadPoolImpl<Task, Tracker, Queue>&& rhs) noexcept
{
    if (this != &rhs)
    {
        m_workers = std::move(rhs.m_workers);
        m_next_worker = rhs.m_next_worker.load();
    }
    return *this;
}

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
template <typename Handler>
inline bool ThreadPoolImpl<Task, Tracker, Queue>::tryPost(Handler &&handler) {
    return getWorker().post(std::forward<Handler>(handler));
}

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
template <typename Handler>
inline void ThreadPoolImpl<Task, Tracker, Queue>::post(Handler &&handler) {
    const auto ok = tryPost(std::forward<Handler>(handler));
    if (!ok) {
        throw std::runtime_error("thread pool queue is full");
    }
}

template<typename Task, template<typename> class Tracker, template<typename> class Queue>
inline Worker<Task, Tracker, Queue> &ThreadPoolImpl<Task, Tracker, Queue>::getWorker() {
    size_t id = Worker<Task, Tracker, Queue>::getWorkerIdForCurrentThread();
    size_t target = m_next_worker.fetch_add(1, std::memory_order_relaxed) % m_workers.size();

    if (target == id) {
        target = (target + 1) % m_workers.size();
    }
    return *m_workers[target];
}
}
