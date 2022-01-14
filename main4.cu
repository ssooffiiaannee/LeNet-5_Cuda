#include <stdio.h>
#include "weights.h"

// __global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int H, int W){
//     int l = blockIdx.y;
//     int c = blockIdx.x;
//     int tid = threadIdx.x;
//     int n_seg = W/(blockDim.x + 1);
    
//     if((l < H) &&  (c < H)){
//         for(int i = 0; i <= n_seg; i++){
//             if(i*blockDim.x + tid >= W)
//                 break;
//             Mout[l*H + c] += M1[l*W + i*blockDim.x + tid] * M2[(tid + i*blockDim.x)*H + c];
//             printf("%f %d %f %f\n",Mout[l*H + c], l*H + c, M1[l*W + i*blockDim.x + tid], M2[(tid + i*blockDim.x)*H + c]);
//         }
//     }
// }

// __global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int H, int W){
//     int l = blockIdx.y;
//     int c = blockIdx.x;
//     for(int i = 0; i<W; i++){
//         Mout[l*H + c] += M1[l*W + i]*M2[c*W + i];
//     }
// }

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int H, int W, int H2){
    int l = blockIdx.y;
    int c = blockIdx.x;
    for(int i = 0; i<W; i++){
        Mout[l*H2 + c] += M1[l*W + i]*M2[c*H2 + i];
//         if(l*H + c == 1)
//             printf("%f %f %d %d %d\n", M1[l*W + i], M2[c*W + i], c, W, i);
    }
//     Mout[l*H2 + c] = (Mout[l*H2 + c]  + b[l*H2 + c]);
}

// __global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int H, int W){
//     int l = blockIdx.y;
//     int c = blockIdx.x;
//     for(int i = 0; i<W; i++){
//         Mout[l*H + c] += M1[l*W + i]*M2[i*H + c];
// //         printf("%f %d %f %f %d %d %d %d\n",Mout[l*H + c], l*H + c, M1[l*W + i], M2[i*H + c], l, c, l*W + i, i*H + l);
//     }
// }

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
            M1[i*W + j] = i * W + j + 1;
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

// void testCudaMatrixMult(int H, int W){
//     float *d_M, *d_Mout;

//     float *M = (float *) malloc(sizeof(float) * H * W);
//     float *Mout = (float *) malloc(sizeof(float) * H * W);
    
//     cudaMalloc(&d_M, sizeof(float)* H * W);
//     cudaMalloc(&d_Mout, sizeof(float)* H * W);

//     for(int i = 0; i<H * W; i++)
//         M[i] = 3;

//     cudaMemcpy(d_M, M, sizeof(float) * H * W, cudaMemcpyHostToDevice);

//     dim3 threadsPerBlock(1024);
//     dim3 blockPerGrid(n, n);
    
//     clock_t begin = clock();
//     cudaMatrixMult<<<blockPerGrid, threadsPerBlock>>>(d_M, d_M, d_Mout, n, n);
// //     cudaMatrixMult<<<blockPerGrid, threadsPerBlock>>>(d_M, d_M, d_Mout, n, n);
//     cudaDeviceSynchronize();
//     clock_t end = clock();
    
//     double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
//     printf("GPU matrix multiplication time for n = %d: %f\n",n , time_spent);
    
//     cudaFree(d_M);
//     cudaFree(d_Mout);
    
//     free(M);
//     free(Mout);
// }

int main(){
    int H = 4, W = 2, H2 = 1;
    float M1[H*W]; // M1*M2
    float M2[H*W];
    float Mout[H*H] = {0};
//     float *Mout = (float *) malloc(sizeof(float) * H * H);
    init(M1, H, W);
    
    print_mat(M1, H, W, 1);
    
    float *M1_d, *M2_d, *Mout_d;
    
    cudaMalloc(&M1_d, sizeof(float)* H * W);
    cudaMalloc(&M2_d, sizeof(float)* H * W);
    cudaMalloc(&Mout_d, sizeof(float)* H * H);
    
    init(M2, H, H2);
//     transpose(M1, M2, H, W);
    print_mat(M2, W, H2, 1);
    
    cudaMemcpy(M1_d, M1, sizeof(float) * H * W, cudaMemcpyHostToDevice);
    cudaMemcpy(M2_d, M2, sizeof(float) * H * H2, cudaMemcpyHostToDevice);
//     cudaMemcpy(Mout_d, Mout, sizeof(float) * H * H, cudaMemcpyHostToDevice);
    
    dim3 n_blocks(H2, H);
    
    cudaMatrixMult<<<n_blocks, 1>>>(M1_d, M2_d, Mout_d, H, W, H2);
    cudaDeviceSynchronize();
    
    cudaMemcpy(Mout, Mout_d, sizeof(float) * H * H2, cudaMemcpyDeviceToHost);
    
//     print_mat(M1, H, W, 1);
//     print_mat(M2, W, H, 1);
    print_mat(Mout, H, H2, 1);
    
    return 0;
}