#include <stdio.h>
//  token : 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


void MatrixInit(float *M, int n, int p){
    srand((unsigned) time(NULL));
    for(int i = 0; i<n; i++){
        for(int j = 0; j < p; j++)
            M[i*p + j] = 2*(rand()/((float)RAND_MAX)) - 1;
    }
}

void MatrixPrint(float *M, int n, int p){
    for(int i = 0; i<n; i++){
        for(int j = 0; j < p; j++)
            printf("%f\n", M[i*p + j]);
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for(int i = 0; i<n; i++){
        for(int j = 0; j < p; j++)
            Mout[i*p + j] = M1[i*p + j]+M2[i*p + j];
    }
}

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < n * p){
        Mout[tid] = M2[tid] + M1[tid];
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    for(int i = 0; i<n; i++){
        for(int j = 0; j < n; j++)
            Mout[i*n + j] = M1[i*n + j]*M2[i*n + j];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < n * p){
        Mout[tid] = M2[tid] * M1[tid];
    }
}

int main(){
    int n = 5, p = 6;
    float *d_M, *d_Mout;
    cudaMalloc(&d_M, sizeof(float)* n * p);
//     cudaMalloc((void**)&d_b, sizeof(float)*N);
    cudaMalloc(&d_Mout, sizeof(float)* n * p);
    
    int n_blocks = 1, n_threads = 256;
    
    
    cudaDeviceSynchronize();
    
    
    float *M = (float *) malloc(sizeof(float) * n * p);
    float *Mout = (float *) malloc(sizeof(float) * n * p);

    MatrixInit(M, n, p);
    cudaMemcpy(d_M, M, sizeof(float) * n * p, cudaMemcpyHostToDevice);
//     cudaMatrixAdd<<<n_blocks, n_threads>>>(d_M, d_M, d_Mout, n, p);
    cudaMatrixMult<<<n_blocks, n_threads>>>(d_M, d_M, d_Mout, n, p);

    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    MatrixPrint(Mout, n, p);
    MatrixPrint(M, n, p);
    
    
//     free(M);
    cudaFree(d_Mout);
    cudaFree(d_M);
    return 0;
}