#ifndef DUMMYMM_H
#define DUMMYMM_H

class DummyMM : public IMulMat
{
public:
    void init(int n, int m, int k,
              int *la, int *lb, int *lc,
              float **A, float **B, float **C) { return };
    void multiply() { return };
};

#endif
