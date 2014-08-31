#include "mymulmat.h"
#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
#include <iostream>
#include <omp.h>
#include <mpi.h>

#define size 4
#define secondsize 2

#define length 16

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
    int i,j, l;
    int k2 = secondsize;
    int m2 = m/secondsize;
    int n2 = n/size;
    
    for(l=0;l<k2;l++){
        for(i=0;i<n2;i++){
            for(j=0;j<m2;j++){
                C[i*m+j+l*m2] = storage_C[l*n2*m2+i*m2+j];
            }
        }
    }
}




void MyMulMat::multiply()
{
    std::cout << "mymul multiply" << std::endl;
    int siz = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();
    int n2 = n/size;
    int m2 = m/secondsize;
    int my_rankA;//初めのn分割したAを配るコミュニケーター
    int my_rankB;

    MPI_Group group_world;
    MPI_Group first_row_group;
    MPI_Comm my_groupA;
    int* process_ranks = (int*)malloc(size*sizeof(int));
    
    for(int proc = 0 ; proc < size ; proc++){
        process_ranks[proc] = proc;
    }
    
    MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    MPI_Group_incl(group_world, size, process_ranks, &first_row_group);
    MPI_Comm_create(MPI_COMM_WORLD, first_row_group, &my_groupA);

    my_rankB = rank % size;    //ブロードキャストされたBをスキャターするコミュニケータ
//4プロセスの場合を想定
    //コミュニケータを作成
    MPI_Comm my_groupB;
    float* tmpC = new float[n*m]();

    if(secondsize > 1){
    
    MPI_Comm_split(MPI_COMM_WORLD, my_rankB, rank, &my_groupB);

    // if(my_rankA==0){
    if(rank < size){
        MPI_Scatter(A, n2*k, MPI_FLOAT, A, n2*k, MPI_FLOAT, 0, my_groupA);
        MPI_Bcast(B, m*k, MPI_FLOAT, 0, my_groupA);
    }
        // }

    if(rank < size ){
        MPI_Barrier(my_groupA);
    }

    float* tB = transpose(B, k, m);

    MPI_Scatter(tB, m2*k, MPI_FLOAT, B, m2*k, MPI_FLOAT, 0, my_groupB);
    MPI_Bcast(A, n2*k, MPI_FLOAT, 0, my_groupB);
 
    for (int i = 0; i < n2; i++){
        for(int j = 0 ; j < m2 ; j+=1){
            for(int l = 0 ; l < k ; l++){ 
                C[i*m2 + j] += A[i*k + l] * B[j*k + l];
            }
        }
    }
    // MPI_Barrier(my_groupB);
    // std::cout << C[0] << std::endl;

    MPI_Gather(C, n2*m2, MPI_FLOAT, tmpC, n2*m2, MPI_FLOAT, 0, my_groupB);
  
    setResult(C, tmpC);
/*
     if(rank == 0 || rank == 1){
        for(int f = 0 ; f < 8 ; f++){
            printf("rank:%d C[%d]:%e\n", rank, f, C[f]);
        }
    }
    */
      if( rank < size ){
//     MPI_Barrier(my_groupA);
    MPI_Gather(C, n2*m, MPI_FLOAT, C, n2*m, MPI_FLOAT, 0, my_groupA);
         }
    //setResult(C, tmpC);

    }else if(secondsize == 1 && size != 1){
        MPI_Scatter(A, n2*k, MPI_FLOAT, A, n2*k, MPI_FLOAT, 0, my_groupA);
        MPI_Bcast(B, m*k, MPI_FLOAT, 0, my_groupA);

        float* tB = transpose(B, k, m);

        for (int i = 0; i < n2; i++){
            for(int j = 0 ; j < m2 ; j+=1){
                for(int l = 0 ; l < k ; l++){ 
                    C[i*m2 + j] += A[i*k + l] * tB[j*k + l];
                }
            }
        }

        MPI_Gather(C, n2*m, MPI_FLOAT, C, n2*m, MPI_FLOAT, 0, my_groupA);
    }else if(secondsize == 1 && size == 1){

        float* tB = transpose(B, k, m);
        for (int i = 0; i < n2; i++){
            for(int j = 0 ; j < m2 ; j+=1){
                for(int l = 0 ; l < k ; l++){ 
                    C[i*m2 + j] += A[i*k + l] * tB[j*k + l];
                }
            }
        }
    }


    return;
}
