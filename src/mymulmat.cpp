#include "mymulmat.h"
#include <iostream>
#include <omp.h>
#include <mpi.h>

MyMulMat::MyMulMat()
{
    std::cout << "mymul constructed" << std::endl;
}

MyMulMat::~MyMulMat()
{
    std::cout << "mymul destructed" << std::endl;
}
void MyMulMat::init(int n, int m, int k,
          int *la, int *lb, int *lc,
          float **A, float **B, float **C)
{
    std::cout << "mymul init" << std::endl;
    *la = k; *lb = m; *lc = m;
    *A = new float[n*k]();
    *B = new float[k*m]();
    *C = new float[n*m]();
    this->n = n; this->m = m; this->k = k;
    this->A = *A; this->B = *B; this->C = *C;
    return;
}

void MyMulMat::multiply()
{
    std::cout << "mymul multiply" << std::endl;
    int size = MPI::COMM_WORLD.Get_size();
//4プロセスの場合を想定
    MPI::COMM_WORLD.Scatter(A, k*n/(size-1), MPI_FLOAT, A, k*n/(size-1), MPI_FLOAT, 0);
    MPI::COMM_WORLD.Bcast(B, m*k, MPI_FLOAT, 0);
    #pragma omp parallel for
    for (int i = 0; i < n/(size-1); i++) {
        for (int j = 0; j < m; j++) {
            for (int l = 0; l < k; l++ ) {
                C[i*m+j] += A[i*k+l] * B[l*m+j];
            }
        }
    }
    MPI::COMM_WORLD.Gather(C, k*n/(size-1), MPI_FLOAT, C, k*n/(size-1), MPI_FLOAT, 0);
    return;
}
