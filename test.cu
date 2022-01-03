#include <stdio.h>
#include <math.h>

struct shorts{
    short a;
    short b;
    short c;
};

struct ints{
    int a;
    int b;
    int c;
};

struct long_longs{
    long long a;
    long long b;
    long long c;
};

__global__ void testing(){
    printf("(%d, %d, %d, %d)\n", blockIdx.y, blockIdx.x, blockDim.x, gridDim.x);
        
//        printf("%d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
//        printf("---\n");
//         printf("%f\n", tanh(0.5));
    
//     printf("%d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
//     printf("---\n");
//     printf("%d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
//     printf("---\n");
//     printf("%d %d %d\n", gridDim.x, gridDim.y, gridDim.z);
}

int main(){
    printf("%ld\n", sizeof(shorts));
    printf("%ld\n", sizeof(ints));
    printf("%ld\n", sizeof(long_longs));
    printf("%ld\n", sizeof(uint3));
    
    int N = 2;
    dim3 threadsPerBlock(1, 1, N);
    dim3 blockPerGrid(20, 20);
    
    testing<<<blockPerGrid, 1>>>();
    
    
//     dim3 threadsPerBlock(N, N);
//     testing<<<1, threadsPerBlock>>>();
    cudaDeviceSynchronize();
//     printf("%f", tanh(0.5));
    return 0;
}