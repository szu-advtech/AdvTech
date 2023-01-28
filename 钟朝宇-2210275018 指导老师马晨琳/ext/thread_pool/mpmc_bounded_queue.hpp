
#pragma once

#include <atomic>
#include <type_traits>
#include <vector>
#include <stdexcept>

namespace tp
{

/**
 * @brief 多生产者/消费之队列的实现
 * 生产者消费者无锁队列。
 * @tparam T 队列元素，只接受可以移动对象。
 */
template <typename T>
class MPMCBoundedQueue
{
    static_assert(
        std::is_move_constructible<T>::value, "Should be of movable type");

public:
     /**
      * @brief 构造函数。
      * @param size 队列长度。
      */
    explicit MPMCBoundedQueue(size_t size);

    /**
     * @brief Move ctor implementation.
     */
    /**
     * @brief 移动构造函数。
     * @param rhs 对象右值引用。
     */
    MPMCBoundedQueue(MPMCBoundedQueue&& rhs) noexcept;

    /**
     * @brief Move assignment implementaion.
     */

    /**
     * @brief 移动赋值函数。
     * @param rhs 对象右值引用。
     * @return 自身引用。
     */
    MPMCBoundedQueue& operator=(MPMCBoundedQueue&& rhs) noexcept;

    /**
     * @brief 将数据放入队列中。
     * @tparam U 数据对象类型。
     * @param data 数据对象。
     * @return 布尔值。
     */
    template <typename U>
    bool push(U&& data);

    /**
     * @brief 将数据弹出队列。
     * @param data 引用，用于接收弹出数据。
     * @return 布尔值。
     */
    bool pop(T& data);

    /**
     * @brief 停止队列。
     */
    void stop();

private:

    /**
     * @brief MPMCBoundedQueue队列元素的存储实现。
     */
    struct Cell {
        /// 原子变量，位置编号。
        std::atomic<size_t> sequence;

        /// 数据。
        T data;

        /**
         * @brief 构造函数。
         */
        Cell() = default;

        /**
         * @brief 拷贝构造函数。
         */
        Cell(const Cell&) = delete;

        /**
         * @brief 赋值构造函数。
         * @return NULL。
         */
        Cell& operator=(const Cell&) = delete;

        /**
         * @brief 移动构造函数。
         * @param rhs 待移动对象的右值。
         */
        Cell(Cell&& rhs): sequence(rhs.sequence.load()), data(std::move(rhs.data)) {
        }

        /**
         * @brief 移动赋值函数。
         * @param rhs 待移动对象的右值。
         * @return 自身引用。
         */
        Cell& operator=(Cell&& rhs) {
            sequence = rhs.sequence.load();
            data = std::move(rhs.data);

            return *this;
        }
    };

private:
    typedef char Cacheline[64];

    /// 数据对齐填充0。
    Cacheline pad0;
    std::vector<Cell> m_buffer;
    /* const */ size_t m_buffer_mask;

    /// 数据对齐填充1。
    Cacheline pad1;
    std::atomic<size_t> m_enqueue_pos;

    /// 数据对齐填充2。
    Cacheline pad2;
    std::atomic<size_t> m_dequeue_pos;

    /// 数据对齐填充3。
    Cacheline pad3;
    size_t tnum;

    /// 数据对齐填充4。
    Cacheline pad4;
};


/// Implementation
template <typename T>
inline void MPMCBoundedQueue<T>::stop() {
}

template <typename T>
inline MPMCBoundedQueue<T>::MPMCBoundedQueue(size_t size)
    : m_buffer(size), m_buffer_mask(size - 1), m_enqueue_pos(0),
      m_dequeue_pos(0)
{
    bool size_is_power_of_2 = (size >= 2) && ((size & (size - 1)) == 0);
    if(!size_is_power_of_2) {
        throw std::invalid_argument("buffer size should be a power of 2");
    }

    for(size_t i = 0; i < size; ++i) {
        m_buffer[i].sequence = i;
    }
}

template <typename T>
inline MPMCBoundedQueue<T>::MPMCBoundedQueue(MPMCBoundedQueue&& rhs) noexcept
{
    *this = rhs;
}

template <typename T>
inline MPMCBoundedQueue<T>& MPMCBoundedQueue<T>::operator=(MPMCBoundedQueue&& rhs) noexcept {
    if (this != &rhs) {
        m_buffer = std::move(rhs.m_buffer);
        m_buffer_mask = std::move(rhs.m_buffer_mask);
        m_enqueue_pos = rhs.m_enqueue_pos.load();
        m_dequeue_pos = rhs.m_dequeue_pos.load();
    }
    return *this;
}

template <typename T>
template <typename U>
inline bool MPMCBoundedQueue<T>::push(U&& data) {
    Cell* cell;
    size_t pos = m_enqueue_pos.load(std::memory_order_relaxed);
    for(;;) {
        cell = &m_buffer[pos & m_buffer_mask];
        size_t seq = cell->sequence.load(std::memory_order_acquire);
        intptr_t dif = (intptr_t)seq - (intptr_t)pos;
        if(dif == 0) {
            if(m_enqueue_pos.compare_exchange_weak(
                   pos, pos + 1, std::memory_order_relaxed)) {
                break;
            }
        }
        else if(dif < 0) {
            return false;
        }
        else {
            pos = m_enqueue_pos.load(std::memory_order_relaxed);
        }
    }

    cell->data = std::forward<U>(data);

    cell->sequence.store(pos + 1, std::memory_order_release);

    return true;
}

template <typename T>
inline bool MPMCBoundedQueue<T>::pop(T& data)
{
    Cell* cell;
    size_t pos = m_dequeue_pos.load(std::memory_order_relaxed);
    for(;;) {
        cell = &m_buffer[pos & m_buffer_mask];
        size_t seq = cell->sequence.load(std::memory_order_acquire);
        intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
        if(dif == 0) {
            if(m_dequeue_pos.compare_exchange_weak(
                   pos, pos + 1, std::memory_order_relaxed)) {
                break;
            }
        }
        else if(dif < 0) {
            return false;
        }
        else {
            pos = m_dequeue_pos.load(std::memory_order_relaxed);
        }
    }

    data = std::move(cell->data);

    cell->sequence.store(
        pos + m_buffer_mask + 1, std::memory_order_release);

    return true;
}

}
