#include "base/detector.hpp"

int main(){
    try{
        Detector detector{"model.tflite"};
    } catch(const std::exception& exp){
        std::cerr << "main: ERROR: " << exp.what() << std::endl;
        return 1;
    }
    return 0;
}