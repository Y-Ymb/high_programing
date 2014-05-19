//openMP無し、1000_1000_1000,ver.1で3.3~3.8GFLOPS。
//openMP無し、1000_1000_1000,ver.2で2.0~2.4GFLOPS。
//openMP無し、1000_1000_1000,ver.3で4.3~4.9GFLOPS。 N=4
//openMP無し、1000_1000_1000,ver.3で5.5~5.6GFLOPS。 N=16
//openMP無し、1000_1000_1000,ver.3で6.4~6.5GFLOPS。 N=32
// 200_200_200 でtransposeに0.452ms 計算に26.294ms　transposeのオーバーヘッドは気にしなくてよさそう
    

#include "mymulmat.h"
#include <iostream>

#include <stdlib.h>
#include <malloc.h>

//#include <immintrin.h>
//#include <xmmintrin.h>

#include <omp.h>
#include <mpi.h>

using namespace std;

MyMulMat::MyMulMat()
{
    std::cout << "mymul constructed" << std::endl;
}

MyMulMat::~MyMulMat()
{
  free(A);
  free(B);
  free(C);
  // free(tB);
  // free(tmp);
  std::cout << "mymul destructed" << std::endl;
}

using std::cout;
using std::endl;

float* MyMulMat::transpose(float *M,int row,int col){
    float *tM = (float*)malloc( sizeof(float) * row * col);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      tM[j*row+i] = M[i*col+j];
    } 
  }
  return tM;
}

void MyMulMat::init(int n, int m, int k,
          int *la, int *lb, int *lc,
          float **A, float **B, float **C)
{
    std::cout << "mymul init" << std::endl;
    /* 横の長さを8の倍数に*/
    *la = k +  (7 - (k-1)%8); 
    *lb = m +  (7 - (m-1)%8); 
    *lc = m +  (7 - (m-1)%8);
    int i, j = 0;
    //プロセス数とプロセス番号をGET
    int rank = MPI::COMM_WORLD.Get_rank();
    int size = MPI::COMM_WORLD.Get_size();

    //rank0のプロセスの場合、行列の面積を確保
//    if( rank == 0 ){
        *A = (float*)malloc( sizeof(float) * n*(*la));
        *B = (float*)malloc( sizeof(float) * k*(*lb));
        *C = (float*)malloc( sizeof(float) * n*(*lc));

    /* 使わない所0埋め。これ意味あるのかな...*/
        for (i = 0; i < n; i++) {
            for (j = k-1; j < *la; j++) {
                (*A)[i*k+j] = 0;
            }
        }
        for (i = 0; i < k; i++) {
            for (j = m-1; j < *lb; j++) {
                (*B)[i*m+j] = 0;
            }
        }
        for (i = 0; i < n; i++) {
            for (j = m-1; j < *lc; j++) {
                (*C)[i*m+j] = 0;
            }
        }
        
    

        cout << "   a % 32 = " << (long)(*A)%32 <<endl;
        cout << "   b % 32 = " << (long)(*B)%32 <<endl;
        cout << "   c % 32 = " << (long)(*C)%32 <<endl;
        this->tB = tB;
        this->tmp = (float*)malloc( sizeof(float) * 8);
    
        this->n = n; this->m = m; this->k = k;
        this->A = *A; this->B = *B; this->C = *C;
//    }
//    else{//rank0以外のプロセスは必要なだけ確保
//        *A = (float*)_mm_malloc( sizeof(float) * k*n/(size-1), 32 );
//        *B = (float*)_mm_malloc( sizeof(float) * k*m/(size-1), 32 );
//        *C = (float*)_mm_malloc( sizeof(float) * n*m/((size-1)*(size-1)), 32 );
//    }
    return;
}

void MyMulMat::multiply()
{
  
    std::cout << "mymul multiply" << std::endl;
    int i, j, l, h=0;
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();
    //rankが0でない場合はIrecvで受け取る
    if(rank != 0){
        MPI::COMM_WORLD.Irecv(A, k*n/(size-1), MPI_FLOAT, 0, 1);
        MPI::COMM_WORLD.Irecv(B, k*m/(size-1), MPI_FLOAT, 0, 2);
    }//rank=0のときは送る
    else{
        float* tB = transpose(B, k, m);
        for( i = 1 ; i < size ; i++){
            MPI::COMM_WORLD.Isend(&A[k*i*n/(size-1)], k*n/(size-1), MPI_FLOAT, i, 1);
            MPI::COMM_WORLD.Isend(&tB[k*i*m/(size-1)], k*m/(size-1), MPI_FLOAT, i, 2);
        }
    }
    printf("ここまでいるよ %d \n", rank);
    if(rank != 0){
        for( j = 0 ; j < n/(size-1) ; j++){
            for( l = 0 ; l < m/(size-1) ; l++){
                for( h = 0 ; h < k ; h++){
                    C[(j*m)/(size-1) + l] += A[j*k+h] + B[l*k+h];
                }
            }
        }
    }
    printf("ここにもいるよw %d \n", rank);
    for( i = 0 ; i < n/(size-1) ; i++ ){
        if(rank != 0){
            MPI::COMM_WORLD.Isend(&C[i*m/(size-1)], m/(size-1), MPI_FLOAT, 0, i);
        }
        else{
            for(j = 1 ; j < size ; j++){
                MPI::COMM_WORLD.Irecv(&C[ i*m + (j-1)*m/(size-1) ], m/(size-1), MPI_FLOAT, j, i);
            }
        }
    }

    
    //tB = transpose(B,k,m);
    //ver.3
    /*
    int N = 32; //ローカルだとNを大きくしてもそんな変わらないけどymmレジスタの数が増えればもっと効果ありそう。
    __m256 VA[N];
    __m256* VB = (__m256*)B;
    __m256* VC = (__m256*)C;
    int i,j,l = 0;
    int m2 = m/8;
    int h = 0;
    for (i = 0; i < n; i+=N) {
      for (j = 0; j < m2; j++) {
	for (l = 0; l < k; l+=1 ) {
	  for(h=0;h<N;h+=1){
	    if(i+h >=n){ //nがNの倍数では無かった場合途中で止める
	      break;
	    }
	    VA[h]= _mm256_broadcast_ss(&A[(i+h)*k+l]);
	    VC[(i+h)*m2+j] = _mm256_add_ps(VC[(i+h)*m2+j], _mm256_mul_ps(VA[h],VB[l*m2+j]));
	  }
	}
      }
    } 
    */
    /*
    //ver.2
    __m256 VA ;
    __m256* VB = (__m256*)B;
    __m256* VC = (__m256*)C;
    int i,j,l = 0;
    int m2 = m/8;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m2; j++) {
            for (l = 0; l < k; l+=1 ) {
	      VA= _mm256_broadcast_ss(&A[i*k+l]);
	      VC[i*m2+j] = _mm256_add_ps(VC[i*m2+j], _mm256_mul_ps(VA,VB[l*m2+j]));
	    }
	}
	} 
    */   
    /*
    //ver.1
    int i,j,l = 0;
    int k2 = k/8;
    __m256* VA = (__m256*)A;
    __m256* VB = (__m256*)tB;
    __m256* VC = (__m256*)C;
  
    {
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            for (l = 0; l < k2; l+=1 ) {
	      //mul_psまでで9.651 [ms]
	      //store までで15.402 [ms]
	      //Cに代入するので20.355 [ms] 0.786048 [GFLOPS]
	      _mm256_mul_ps(VA[i*k2+l],VB[j*k2+l]);
	      _mm256_store_ps(tmp,_mm256_mul_ps(VA[i*k2+l],VB[j*k2+l])); 
	      C[i*m+j] += tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7] ;
	      //ここの処理もリダクション出来ないか？512では_mm512_reduce_add_psがあるんだけど
	    }
        }
      }
    }
    */
    free(tB);
    return;
}
