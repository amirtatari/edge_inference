#pragma once

#include <chrono>
#include <unordered_map>
#include <mutex>

namespace profiling
{
  
#define PROFILE_FUNCTION() profiler::Profiler _p_{__func__}
  
using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

};

struct Result
{
  long long m_totalTime {0};
  int m_numCalls {0};

  double calculateAverageTime() const;
};

struct Storage
{
  static void addData(const char* funcName, long long duration);

  static void printSummary();

  static std::unordered_map<const char*, Result> m_results;
  static std::mutex m_mtx;
};

struct Profiler
{
  explicit Profiler(const char* funcName) noexcept;

  ~Profiler();

private:
  profiling::TimePoint m_start;
  profiling::TimePoint m_end;
  const char* m_funcName;
};

inline std::unordered_map<const char*, Result> Storage::m_results;
inline std::mutex Storage::m_mtx;
