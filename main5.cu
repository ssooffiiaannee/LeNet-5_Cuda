#include <stdio.h>
#include "weights.h"

__global__ void cudaMatrixMult(float *Mout, int H, int W){
    printf("%d %d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
//     int tid = threadIdx.x;
//     if(tid < 9)
//         Mout[tid] += 1;
    
//     printf("%d %d %d\n", tid, blockIdx.x, blockIdx.y);
//     for(int i = 0; i<H; i++){
//         for(int j = 0; j<H; j++)
//             Mout = 1;
//     }
}

void transpose(float *in, float *out, int H, int W){
    for(int i = 0; i<H; i++){
        for(int j = 0; j<W; j++){
            out[j*H + i] = in[i*W + j];
        }
    }
}

void init(float *M1, int H, int W){
    for(int i = 0; i<H; i++){
        for(int j = 0; j<W; j++)
            M1[i*W + j] = i * W + j;
    }
}

void print_mat(float *vec, int H, int W, int n_c){
    for(int l = 0; l<n_c; l++){
        for(int i = 0; i<H; i++){
            for(int j = 0; j<W; j++)
//                 printf("%d ", 7*((int) ceil(vec[l*L*H + i*L + j])));
                printf("%1.6f ", vec[l*W*H + i*W + j]);
            printf("\n");
        }
        printf("\n");
    }
}


int main(){
    int H = 3, W = 2;
    float *Mout = (float *) malloc(sizeof(float) * H * H);
    float *Mout_d;
    cudaMalloc(&Mout_d, sizeof(float)* H * H);
    
    
    dim3 n_blocks(4, 3);
    dim3 threadPerBlock(33, 32);
    
    cudaMatrixMult<<<n_blocks, threadPerBlock>>>(Mout_d, H, W);
//     cudaDeviceSynchronize();
    
    cudaMemcpy(Mout, Mout_d, sizeof(float) * H * H, cudaMemcpyDeviceToHost);

    print_mat(Mout, H, H, 1);
    
    return 0;
}