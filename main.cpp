#include "base/engine.hpp"

#include <iostream>

int main()
{
    try
    {
        Engine engine{"/path/to/config/file"};
    } 
    catch(const std::exception& exp)
    {
        std::cerr << "main: ERROR: " << exp.what() << std::endl;
        return 1;
    }
    return 0;
}