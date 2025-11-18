#include "profiler.h"

#include <iostream>

Profiler::Profiler(const char* funcName) noexcept 
  : m_start {profiling::Clock::now()}
  , m_end {}
  , m_funcName {funcName}
{}

Profiler::~Profiler()
{
  m_end = profiling::Clock::now();
  auto duration {std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start)};
  Storage::addData(m_funcName, duration.count());
}

double Result::calculateAverageTime() const
{
  if (m_numCalls == 0)
  {
    return 0.0;
  }

  return m_totalTime / (double)m_numCalls;
}

void Storage::printSummary()
{
  std::cout << "--- Profiler Summary ---\n";
  std::cout.precision(4); // Set decimal places
  std::cout << std::fixed;

  for (const auto& [name, result] : m_results)
  {
    std::cout << name << ": \n"
	      << "  Calls: " << result.m_numCalls << "\n"
	      << "  Avg:   " << result.calculateAverageTime() << " ms\n"
	      << "  Total: " << (result.m_totalTime) << " ms\n";
  } 
}

void Storage::addData(const char* funcName, long long duration)
{
  m_results[funcName].m_totalTime += duration;
  m_results[funcName].m_numCalls += 1;
}
