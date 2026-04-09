#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>

/**
 * @brief thread safe implementation of the queue
 * @tparam T type of the item that queue holds
 */
template<typename T>
class Queue
{
    std::queue<T> m_queue;
    std::mutex m_mtx;
    std::condition_variable m_cv;
  
  public:  
    Queue() = default;
    Queue(Queue&&) = delete;
    Queue(const Queue&) = delete;
    Queue& operator=(Queue&&) = delete;
    Queue& operator=(const Queue&) = delete;
  
    /**
     * @brief push the item in queue
     * @param item of type T
     */
    void push(const T& item);
    void push(T&& item);

    /**
     * @brief receives the front item in queue
     * @return front item
     */
    T pop();
};

template<typename T>
void Queue<T>::push(const T& item)
{
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_queue.push(item);
    }
    m_cv.notify_one();
}

template<typename T>
void Queue<T>::push(T&& item)
{
    {
        std::lock_guard<std::mutex> lock(m_mtx);
        m_queue.push(std::move(item));
    }
    m_cv.notify_one();
}

template<typename T>
T Queue<T>::pop()
{
    std::unique_lock<std::mutex> lock(m_mtx);
    m_cv.wait(lock, [this]{ return !m_queue.empty(); });
    auto item {std::move(m_queue.front())};
    m_queue.pop();
    return item;
}