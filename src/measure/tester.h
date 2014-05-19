#pragma once

#include <sstream>
#include <iostream>
#include "dataset.h"

class Tester
{
public:
    Tester();
    ~Tester();
    void run();
    void run(int argc, char *argv[]);
private:
    void _run(Dataset::DataType);
    void print(std::string str);
    uint64_t getus();
    int myrank;
};
