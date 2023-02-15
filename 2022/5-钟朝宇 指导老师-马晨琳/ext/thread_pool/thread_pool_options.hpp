#pragma once

#include <algorithm>
#include <thread>

namespace tp
{

/**
 * @brief The ThreadPoolOptions class provides creation options for
 * ThreadPool.
 */
class ThreadPoolOptions
{
public:
    /**
     * @brief ThreadPoolOptions Construct default options for thread pool.
     */
    ThreadPoolOptions(int i, bool prefer = false);

    /**
     * @brief setThreadCount Set thread count.
     * @param count Number of threads to be created.
     */
    void setThreadCount(size_t count);

    /**
     * @brief setQueueSize Set single worker queue size.
     * @param size Maximum length of queue of single worker.
     */
    void setQueueSize(size_t size);

    /**
     * @brief threadCount Return thread count.
     */
    size_t threadCount() const;

    /**
     * @brief queueSize Return single worker queue size.
     */
    size_t queueSize() const;

    void setIOPreference(bool flag);

    bool isIOPreference() const {
        return io_preference;
    }
private:
    size_t m_thread_count;
    size_t m_queue_size;
    bool io_preference = false;
};

/// Implementation

inline ThreadPoolOptions::ThreadPoolOptions(int i, bool prefer)
    : m_thread_count(std::min<size_t>(std::max<size_t>(1u, std::thread::hardware_concurrency()), i)),
      m_queue_size(8192u),
      io_preference(prefer)
{
}

inline void ThreadPoolOptions::setThreadCount(size_t count)
{
    m_thread_count = std::max<size_t>(1u, count);
}

inline void ThreadPoolOptions::setIOPreference(bool flag)
{
    io_preference = flag;
}

inline void ThreadPoolOptions::setQueueSize(size_t size)
{
    m_queue_size = std::max<size_t>(1u, size);
}

inline size_t ThreadPoolOptions::threadCount() const
{
    return m_thread_count;
}

inline size_t ThreadPoolOptions::queueSize() const
{
    return m_queue_size;
}

}
