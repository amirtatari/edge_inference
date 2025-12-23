#include "testBench/testBench.h"

int main()
{
  TestBenchFactory tbfactory;
  tbfactory.start("path/to/config");
  return 0;
}
