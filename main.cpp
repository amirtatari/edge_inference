#include "testBench/testBench.h"
#include <spdlog/spdlog.h>

int main(int argc, char *argv[])
{
  if (argc == 3 && std::string(argv[1]) == "--config")
  {
    const std::string configPath {argv[2]};
    TestBenchFactory tbfactory;
    tbfactory.start(configPath);
  }
  else
  {
    spdlog::error("Usage: {} --config <path/to/config.xml>", argv[0]);
    return -1;
  }
  return 0;
}
