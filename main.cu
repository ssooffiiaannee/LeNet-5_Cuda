#include <stdio.h>
#include <time.h>
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
            printf("%1.1f ", M[i*p + j]);
        printf("\n");
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
        for(int j = 0; j<n; j++){
            for(int k = 0; k<n; k++){
                Mout[n*j + i] = M1[j*n + k] * M2[i + k*n];
            }
        }
    }
}

void testMatrixMult(){
    int n = 1000;
    float *M = (float *) malloc(sizeof(float) * n * n);
    float *Mout = (float *) malloc(sizeof(float) * n * n);
    
    for(int i = 0; i<n*n; i++)
        M[i] = 3;
    
    clock_t begin = clock();
    MatrixMult(M, M, Mout, n);
    clock_t end = clock();    
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("cpu matrix multiplication time for n = %d: %f\n",n , time_spent);
    
//     MatrixPrint(Mout, n, n);
    
    free(M);
    free(Mout);
}

// __global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n_l, int n_c){
//     int l = blockIdx.y;
//     int c = blockIdx.x;
//     int i = 0;
// //     printf("%d\n", gridDim.x);
//     for(; i < n_c; i++){
//         Mout[l*n_c + c] += M1[l*n_c + i] * M2[i*n_c + c];
//     }
// }


__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n_l, int n_c){
    int l = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int seg_sz = n_c/blockDim.x + 1;
    
    for(int i = 0; i < seg_sz; i++){
        if(i*blockDim.x + tid >= n_c)
            break;
        Mout[l*n_c + c] += M1[l*n_c + i*blockDim.x + tid] * M2[(tid + i*blockDim.x)*n_c + c];
    }
}

void testCudaMatrixAdd(){
    int n = 5, p = 1;
    float *d_M, *d_Mout;
    cudaMalloc(&d_M, sizeof(float)* n * p);
    cudaMalloc(&d_Mout, sizeof(float)* n * p);
    
    int n_blocks = 1, n_threads = 256;
    
    float *M = (float *) malloc(sizeof(float) * n * p);
    float *Mout = (float *) malloc(sizeof(float) * n * p);

    MatrixInit(M, n, p);
    cudaMemcpy(d_M, M, sizeof(float) * n * p, cudaMemcpyHostToDevice);
    
    cudaMatrixAdd<<<n_blocks, n_threads>>>(d_M, d_M, d_Mout, n, p);

    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * p, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    MatrixPrint(M, n, p);
    printf("\n");
    MatrixPrint(Mout, n, p);
    
    free(M);
    
    cudaFree(d_Mout);
    cudaFree(d_M);
}

void testCudaMatrixMult(){
    int n = 1000;
    float *d_M, *d_Mout;

    float *M = (float *) malloc(sizeof(float) * n * n);
    float *Mout = (float *) malloc(sizeof(float) * n * n);
    
    cudaMalloc(&d_M, sizeof(float)* n * n);
    cudaMalloc(&d_Mout, sizeof(float)* n * n);

    for(int i = 0; i<n*n; i++)
        M[i] = 3;

    cudaMemcpy(d_M, M, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024);
    dim3 blockPerGrid(n, n);
    
    clock_t begin = clock();
    cudaMatrixMult<<<blockPerGrid, threadsPerBlock>>>(d_M, d_M, d_Mout, n, n);
//     cudaMatrixMult<<<blockPerGrid, threadsPerBlock>>>(d_M, d_M, d_Mout, n, n);
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("GPU matrix multiplication time for n = %d: %f\n",n , time_spent);
    
//     cudaMemcpy(Mout, d_Mout, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    
    
//     MatrixPrint(M, n, n);
//     printf("\n");
//     MatrixPrint(Mout, n, n);
    
    cudaFree(d_M);
    cudaFree(d_Mout);
    
    free(M);
    free(Mout);
}

int main(){
//     testCudaMatrixAdd();
    testCudaMatrixMult();
    testMatrixMult();
    return 0;
}