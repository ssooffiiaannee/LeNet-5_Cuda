#include <stdio.h>
#include <time.h>


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

void print_mat(float *vec, int L, int H, int n_c){
    for(int l = 0; l<n_c; l++){
        for(int i = 0; i<H; i++){
            for(int j = 0; j<L; j++)
//                 printf("%d ", 7*((int) ceil(vec[l*L*H + i*L + j])));
                printf("%1.6f ", vec[l*L*H + i*L + j]);
            printf("\n");
        }
        printf("\n");
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

double testMatrixMult(int mat_size){
    int n = mat_size;
    float *M = (float *) malloc(sizeof(float) * n * n);
    float *Mout = (float *) malloc(sizeof(float) * n * n);
    
    for(int i = 0; i<n*n; i++)
        M[i] = 3;
    
    clock_t begin = clock();
    MatrixMult(M, M, Mout, n);
    clock_t end = clock();    
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("cpu matrix multiplication time for n = %d: %f\n",n , time_spent);
    
    free(M);
    free(Mout);
    
    return time_spent;
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
__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n_l, int n_c){
    int c = blockIdx.x*blockDim.x + threadIdx.x;
    int l = blockDim.y*blockIdx.y + threadIdx.y;
    
    if(l < n_c && c < n_c)
        for(int i = 0; i < n_c; i++){
            Mout[l*n_l + c] += M1[l*n_c + i]*M2[i*n_l + c];
        }
}


double testCudaMatrixMult(int mat_size){
    int n = mat_size;
    float *d_M, *d_Mout;

    float *M = (float *) malloc(sizeof(float) * n * n);
    float *Mout = (float *) malloc(sizeof(float) * n * n);
    
    cudaMalloc(&d_M, sizeof(float)* n * n);
    cudaMalloc(&d_Mout, sizeof(float)* n * n);

    for(int i = 0; i<n*n; i++)
        M[i] = 3;

    cudaMemcpy(d_M, M, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blockPerGrid(n, n);
    
    clock_t begin = clock();
    cudaMatrixMult<<<blockPerGrid, threadsPerBlock>>>(d_M, d_M, d_Mout, n, n);
    cudaDeviceSynchronize();
    clock_t end = clock();
    
    cudaMemcpy(Mout, d_Mout, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("GPU matrix multiplication time for n = %d: %f seconds\n",n , time_spent);
//     print_mat(Mout, n, n, 1);
    printf("%f %f\n", Mout[0], Mout[n*n - 1]);
    cudaFree(d_M);
    cudaFree(d_Mout);
    
    free(M);
    free(Mout);
    
    return time_spent;
}

int main(){
    int n;
    printf("######## n x n matrix multiplication #######\n");
    printf("Enter matrix size n : \n");
    scanf("%d", &n);
    printf("GPU computation running ...\n");
    double t1 = testCudaMatrixMult(n);
    printf("CPU computation running ...\n");
    double t2 = testMatrixMult(n);
    printf("GPU is %d times faster.\n", (int) (t2/t1));
    return 0;
}
