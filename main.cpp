#include "testBench/testBench.hpp"

#include <spdlog/spdlog.h>

int main(int argc, char *argv[])
{
  if (argc == 3 && std::string(argv[1]) == "--config")
  {
    const std::string configPath {argv[2]};
    TaskFactory taskFactory(configPath);
  }
  else
  {
    spdlog::error("Usage: {} --config <path/to/config.xml>", argv[0]);
    return -1;
  }
  return 0;
}
