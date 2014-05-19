#pragma once
#include "measure/imulmat.h"
class MyMulMat : public IMulMat
{
public:
    MyMulMat();
    ~MyMulMat();
    void init(int n, int m, int k,
              int *la, int *lb, int *lc,
              float **A, float **B, float **C);
    void multiply();
private:
    int n, m, k;
    float *A, *B, *C;
};

