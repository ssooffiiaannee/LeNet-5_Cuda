#include <stdio.h>
#include "weights.h"


// nvcc compilation not working when jupyter-lab kernel is running

__global__ void Conv2d(float *image, float *ker, float *out, float *biases, int im_size, int k_size, int n_ker, int n_c){    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int b_id = blockIdx.x;
    int out_size = im_size - k_size + 1;

    for(int i = 0; i < n_c; i++){
        for(int j = 0; j<k_size; j++){
            for(int k = 0; k<k_size; k++){
                int ker_co = i*k_size*k_size + b_id*k_size*k_size*n_c + k_size*j + k;
                int im_co = i*im_size*im_size + (j+tid_y)*im_size + k + tid_x;
                out[b_id*out_size*out_size + tid_y*(out_size) + tid_x] += image[im_co] * ker[ker_co];
//                 printf("%f \n", out[b_id*out_size*out_size + tid_y*(out_size) + tid_x]);
//                 if(image[im_co] != 0)
//                     printf("%d %d %d %f %f \n", im_co, (j+tid_y), k + tid_x, image[im_co], ker[ker_co]);
            }
        }
    }
//     printf("%f\n", out[b_id*out_size*out_size + tid_y*(out_size) + tid_x]);
    out[b_id*out_size*out_size + tid_y*(out_size) + tid_x] = tanh(out[b_id*out_size*out_size + tid_y*(out_size) + tid_x] + biases[b_id]);
//     out[b_id*out_size*out_size + tid_y*(out_size) + tid_x] = out[b_id*out_size*out_size + tid_y*(out_size) + tid_x];// + biases[b_id];
//     printf("%f\n", out[b_id*out_size*out_size + tid_y*(out_size) + tid_x]);

}

// __global__ void Conv2d(){
//     printf("sdj\n");
// }

__global__ void averagePool(float *in, int len_x, int len_y, int len_z, float *out){
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int b_id = blockIdx.x;
    int surf = len_x*len_y;
//     printf("%d %d %d\n", tid_x, tid_y, b_id);
    out[surf/4*b_id + tid_y*len_x/2 + tid_x] = (in[surf*b_id + tid_y*len_x*2 + tid_x*2] + 
                                                in[surf*b_id + tid_y*len_x*2 + tid_x*2 + 1] + 
                                                in[surf*b_id + tid_y*len_x*2 + tid_x*2 + len_x] + 
                                                in[surf*b_id + tid_y*len_x*2 + tid_x*2 + len_x + 1])/4;
//     printf("%f\n", in[tid_x]);
}

__device__ float activation_tanh(float M){
    return tanh(M);
}

__device__ float activation_softmax(float *vec, int n, int i){
    float sum = 0;
    
    for(int j = 0; j<n; j++){
        sum += exp(vec[j]);
    }
    
    return exp(vec[i])/sum;
}

void init(float *image, int L, int n_ker){
    for(int l = 0; l < n_ker; l++){
        for(int i = 0; i<L; i++){
            for(int j = 0; j<L; j++){
                image[l*L*L + i*L + j] = i*L + j;//i;//i + j + 2*l;
            }
        }
    }
}


void print_vec(float *vec, int L){
    for(int i = 0; i<L; i++){
        printf("%f\n", vec[i]);
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


float * Conv1(float *im, int H, int W, float *ker, float *biases, int n_ker, int n_c){
    int k_size = 5;
    int k_vol = k_size * k_size * n_ker;
    int out_vol = (W-k_size + 1)*(H-k_size + 1)*n_ker;
    int img_vol = H * W * n_c;
    
//     float *kernel = (float *) malloc(sizeof(float) k_vol);
    float *out = (float *) malloc(sizeof(float) * out_vol);
                                     
    float *im_d, *out_d, *ker_d, *biases_d;
    cudaMalloc(&im_d, sizeof(float) * img_vol);
    cudaMalloc(&out_d, sizeof(float) * out_vol);
    cudaMalloc(&ker_d, sizeof(float) * k_vol);
    cudaMalloc(&biases_d, sizeof(float) * n_ker);
    
    cudaMemcpy(im_d, im, sizeof(float) * img_vol, cudaMemcpyHostToDevice);
    cudaMemcpy(ker_d, ker, sizeof(float) * k_vol, cudaMemcpyHostToDevice);
    cudaMemcpy(biases_d, biases, sizeof(float) * n_ker, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock((W-k_size + 1), (H-k_size + 1));
    dim3 blockPerGrid(n_ker);
    printf("################ Conv1 ########################\n\n");
//     Conv2d<<<blockPerGrid, threadPerBlock>>>();
    Conv2d<<<blockPerGrid, threadPerBlock>>>(im_d, ker_d, out_d, biases_d, H, k_size, n_ker, n_c);
    cudaDeviceSynchronize();

    cudaMemcpy(out, out_d, sizeof(float) * out_vol, cudaMemcpyDeviceToHost);
    
//     print_mat(out, (W-k_size + 1), (H-k_size + 1), n_ker);
    
    cudaFree(im_d);
    cudaFree(out_d);
    cudaFree(ker_d);
    cudaFree(biases_d);
    
//     free(out);
    return out;
}

float * MAP1(float *in, unsigned int W, unsigned int H, int n_c){
    float *outMAP_d,  *in_d;
    int vol_out_MAP = W * H * n_c / 4;
    int in_vol = W * H * n_c;
    
    float *outMAP = (float *) malloc(sizeof(float) * vol_out_MAP);
    cudaMalloc(&outMAP_d, sizeof(float) * vol_out_MAP);
    cudaMalloc(&in_d, sizeof(float) * in_vol);
    cudaMemcpy(in_d, in, sizeof(float) * in_vol, cudaMemcpyHostToDevice);

    averagePool<<<n_c, { W/2, H/2}>>>(in_d, W, H, n_c, outMAP_d);
    cudaDeviceSynchronize();
    cudaMemcpy(outMAP, outMAP_d, sizeof(float) * vol_out_MAP, cudaMemcpyDeviceToHost);
    
    printf("################ MAP1 ########################\n\n");
//     print_mat(outMAP, W/2, H/2, n_c);
//     print_mat(in, W, H, n_c);
    
    cudaFree(outMAP_d);
    
//     free(outMAP);
    return outMAP;
}

void zero_padding(float *im, int H, int pad, float *out){
    for(int i = 0; i<H + 2*pad; i++){
        for(int j = 0; j<H + 2*pad; j++){
            if((i < pad) || (j < pad) || (i >= H + pad) || (j >= H + pad)){
                out[i*(H + 2*pad) + j] = 0;
                continue;
            }
//             if(abs(im[(i - pad)*H + j - pad] - 0.996) < 0.001) printf("%d %d %f\n", i, j, im[(i - pad)*H + j - pad]);
            out[i*(H + 2*pad) + j] = im[(i - pad)*H + j - pad];
        }
    }
}

int main(){
    int H = 5, pad = 0, n_c = 2;
    int H_p = H + 2*pad;
    float im[H * H * n_c];
    float im_p[H_p * H_p];

    init(im, H, 1);
    zero_padding(im, H, pad, im_p);
    
//     print_mat(im_p, H_p, H_p , 1);
    
//     float *l1;
    for(int i = 0; i<2400; i++)
        printf("%f\n", conv2d_1_w[i]);
    
    
    
//     l1 = Conv1(im_p, H_p, H_p, conv2d_w, conv2d_b, 6, 1);
//     l1 = MAP1(l1, H, H, 6);
    
// //     print_mat(l1, H/2, H/2, 6);
//     l1 = Conv1(l1, H/2, H/2, conv2d_1_w, conv2d_1_b, 16, 6);
//     print_mat(l1, H/2, H/2, 16);
// //     l1 = MAP1(l1, H/2 - 4, H/2 - 4, 16);
// //     print_mat(l1, (H/2 - 4)/2, (H/2 - 4)/2, 16);
    
//     free(l1);
    return 0;
}

