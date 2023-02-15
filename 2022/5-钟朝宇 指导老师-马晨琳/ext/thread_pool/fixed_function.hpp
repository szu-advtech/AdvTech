#pragma once

#include <type_traits>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace tp
{

/**
 * @brief FixedFunction<R(ARGS...), STORAGE_SIZE> 类型实现了函数对象。
 * 这个函数对象是对std::function的模仿，但它的容量是受限的：
 *  - 它只支持移动语义
 *  - 它的函数对象大小受限于存储容量（STORAGE_SIZE）
 * 由于上述限制，它在创建和拷贝时快于std::function
 */
template <typename SIGNATURE, size_t STORAGE_SIZE = 128>
class FixedFunction;

/**
 * @brief 函数对象的封装。
 * @tparam R 函数返回值类型。
 * @tparam ARGS 形式参数类型。
 * @tparam STORAGE_SIZE 存储容量。
 */
template <typename R, typename... ARGS, size_t STORAGE_SIZE>
class FixedFunction<R(ARGS...), STORAGE_SIZE>
{
    /// 定义R(ARGS...)类型的函数指针func_ptr_type
    typedef R (*func_ptr_type)(ARGS...);

public:
    /**
     * @berief FixedFunction构造函数。
     */
    FixedFunction(): m_function_ptr(nullptr), m_method_ptr(nullptr),
          m_alloc_ptr(nullptr) {
    }

    /**
     * @brief FixedFunction构造函数。根据函数对象构造FixedFunction。
     * @param object 函数对象会使用移动构造的方法存储在内部存储单元，不可移动的函数对象是被禁止的。
     */
    template <typename FUNC>
    FixedFunction(FUNC&& object): FixedFunction() {
        typedef typename std::remove_reference<FUNC>::type unref_type;

        static_assert(sizeof(unref_type) < STORAGE_SIZE,
            "functional object doesn't fit into internal storage");
        static_assert(std::is_move_constructible<unref_type>::value,
            "Should be of movable type");

        m_method_ptr = [](
            void* object_ptr, func_ptr_type, ARGS... args) -> R
        {
            return static_cast<unref_type*>(object_ptr)->operator()(args...);
        };

        m_alloc_ptr = [](void* storage_ptr, void* object_ptr)
        {
            if(object_ptr) {
                unref_type* x_object = static_cast<unref_type*>(object_ptr);
                new(storage_ptr) unref_type(std::move(*x_object));
            }
            else {
                static_cast<unref_type*>(storage_ptr)->~unref_type();
            }
        };

        m_alloc_ptr(&m_storage, &object);
    }

    /**
     * @brief FixedFunction的构造函数，用于对函数指针进行封装。
     * @tparam RET 函数返回值类型。
     * @tparam PARAMS 函数形式参数类型。
     * @param func_ptr 函数指针。
     */
    template <typename RET, typename... PARAMS>
    FixedFunction(RET (*func_ptr)(PARAMS...)): FixedFunction()
    {
        m_function_ptr = func_ptr;
        m_method_ptr = [](void*, func_ptr_type f_ptr, ARGS... args) -> R
        {
            return static_cast<RET (*)(PARAMS...)>(f_ptr)(args...);
        };
    }

    /**
     * @brief 移动构造函数。
     * @param o 待移动对象的右值。
     */
    FixedFunction(FixedFunction&& o) : FixedFunction() {
        moveFromOther(o);
    }

    /**
     * @brief 重载移动赋值运算符。
     * @param o 待移动对象的右值。
     * @return 自身的引用。
     */
    FixedFunction& operator=(FixedFunction&& o) {
        moveFromOther(o);
        return *this;
    }

    /**
     * @brief 析构函数，同时调用函数对象的析构函数。
     */
    ~FixedFunction() {
        if(m_alloc_ptr) m_alloc_ptr(&m_storage, nullptr);
    }

    /**
     * @brief 执行所存储的函数对象。
     * @throws 如果函数对象不存在会抛出运行时错误。
     */
    R operator()(ARGS... args)
    {
        if(!m_method_ptr) throw std::runtime_error("call of empty functor");
        return m_method_ptr(&m_storage, m_function_ptr, args...);
    }

private:
    /**
     * @brief 删除拷贝赋值函数。
     * @return NULL。
     */
    FixedFunction& operator=(const FixedFunction&) = delete;

    /**
     * @brief 删除拷贝构造函数。
     */
    FixedFunction(const FixedFunction&) = delete;

    union
    {
        /// 固定的内存空间，用于存放函数对象。
        typename std::aligned_storage<STORAGE_SIZE, sizeof(size_t)>::type m_storage;

        /// 函数指针。
        func_ptr_type m_function_ptr;
    };


    typedef R (*method_type)(void* object_ptr, func_ptr_type free_func_ptr, ARGS... args);
    typedef void (*alloc_type)(void* storage_ptr, void* object_ptr);

    /// 内存空间分配函数的指针。
    alloc_type m_alloc_ptr;

    /// 函数对象封装指针。
    method_type m_method_ptr;

    /**
     * @brief 将其他FixedFunction对象的成员变量转移到当前对象。
     * @param o 对象的引用。
     */
    void moveFromOther(FixedFunction& o) {
        if(this == &o) return;

        if(m_alloc_ptr) {
            m_alloc_ptr(&m_storage, nullptr);
            m_alloc_ptr = nullptr;
        }
        else {
            m_function_ptr = nullptr;
        }

        m_method_ptr = o.m_method_ptr;
        o.m_method_ptr = nullptr;

        if(o.m_alloc_ptr) {
            m_alloc_ptr = o.m_alloc_ptr;
            m_alloc_ptr(&m_storage, &o.m_storage);
        }
        else {
            m_function_ptr = o.m_function_ptr;
        }
    }
};

}
