#include "../utils/queue/queue.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>
#include <memory>

TEST(QueueTest, PushPopSingleThread) 
{
    Queue<int> q;
    q.push(1);
    q.push(2);

    EXPECT_EQ(q.pop(), 1);
    EXPECT_EQ(q.pop(), 2);
}

TEST(QueueTest, HandlesMoveOnlyTypes) 
{
    Queue<std::unique_ptr<int>> q;
    auto ptr = std::make_unique<int>(50);
    
    q.push(std::move(ptr));
    
    auto result = q.pop();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(*result, 50);
}

TEST(QueueTest, ConcurrentPushPop) 
{
    Queue<int> q;
    constexpr int num_threads {5};
    constexpr int items_per_thread {1000};
    std::atomic<int> sum{0};

    std::vector<std::jthread> producers;
    std::vector<std::jthread> consumers;

    // producers
    for (int i = 0; i < num_threads; ++i) {
        producers.emplace_back([&q, items_per_thread] {
            for (int j = 0; j < items_per_thread; ++j) {
                q.push(1);
            }
        });
    }

    // consumers
    for (int i = 0; i < num_threads; ++i) {
        consumers.emplace_back([&q, items_per_thread, &sum] {
            for (int j = 0; j < items_per_thread; ++j) {
                sum += q.pop();
            }
        });
    }

    producers.clear(); 
    consumers.clear();

    EXPECT_EQ(sum.load(), num_threads * items_per_thread);
}