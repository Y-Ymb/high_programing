#pragma once

class IMulMat {
public:
    virtual void init(int n, int m, int k,
                      int *la, int *lb, int *lc,
                      float **A, float **B, float **C) = 0;
    virtual void multiply() = 0;
};
