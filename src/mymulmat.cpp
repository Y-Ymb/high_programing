#include "mymulmat.h"
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
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

float* MyMulMat::transpose(float *M, int row, int col){
    float *tM = (float*)malloc(sizeof(float)*row*col);
    for(int i = 0 ; i < row; i++){
        for(int j = 0 ; j < col; j++){
            tM[j*row+i] = M[i*col+j];
        }
    }
    return tM;
}

void MyMulMat::setResult(float *C,float *storage_C){
    int i,j;
    int m2 = m/4;
    int n2 = n/4;
    for(i=0;i<n2;i++){
        for(j=0;j<m2;j++){
            C[i*m+j] = storage_C[i*m2+j];
        }
    }
}





void MyMulMat::multiply()
{
    std::cout << "mymul multiply" << std::endl;
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();
    
    int my_rankA;//初めのn分割したAを配るコミュニケーター
    int my_rankB;

    MPI_Group group_world;
    MPI_Group first_row_group;
    MPI_Comm my_groupA;
    int* process_ranks = (int*)malloc(5*sizeof(int));
    
    for(int proc = 0 ; proc < 4 ; proc++){
        process_ranks[proc] = proc;
    }
    
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    MPI_Group_incl(group_world, 5, process_ranks, &first_row_group);
    MPI_Comm_create(MPI_COMM_WORLD, first_row_group, &my_groupA);

    my_rankB = rank % 4;    //ブロードキャストされたBをスキャターするコミュニケータ
//4プロセスの場合を想定
    //コミュニケータを作成
    MPI_Comm my_groupB;
    float* tmpC = new float[n*m]();
    
    MPI_Comm_split(MPI_COMM_WORLD, my_rankB, rank, &my_groupB);

    // if(my_rankA==0){
    if(rank == 0 || rank == 1 || rank == 2 || rank == 3){
        MPI_Scatter(A, n*k/4, MPI_FLOAT, A, n*k/4, MPI_FLOAT, 0, my_groupA);
        MPI_Bcast(B, m*k, MPI_FLOAT, 0, my_groupA);
    }
        // }

    float* tB = transpose(B, k, m);

    MPI_Scatter(tB, m*k/4, MPI_FLOAT, B, m*k/4, MPI_FLOAT, 0, my_groupB);
    MPI_Bcast(A, n*k/4, MPI_FLOAT, 0, my_groupB);
 
    for (int i = 0; i < n/4; i++){
        for(int j = 0 ; j < m/4 ; j++){
            for(int l = 0 ; l < k ; l++){ 
                C[i*m/4+j] += A[i*k+l] * B[j*k+l];
            }
        }
    }
    // std::cout << C[0] << std::endl;
    
    
    MPI_Gather(C, n*m/16, MPI_FLOAT, C, n*m/16, MPI_FLOAT, 0, my_groupB);
    //setResult(C, tmpC);
    
    if(rank == 0 || rank == 1 || rank == 2 || rank == 3 ){
        MPI_Gather(C, m, MPI_FLOAT, C, m, MPI_FLOAT, 0, my_groupA);
    }
    // setResult(C, tmpC);
    return;
}
