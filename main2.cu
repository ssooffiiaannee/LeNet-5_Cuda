#include <stdio.h>

__global__ void conv(float *image, float *ker, float *out, int im_size, int k_size, int n_ker){
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int b_id = blockIdx.x;
    
        for(int j = 0; j<k_size; j++){
            for(int k = 0; k<k_size; k++){
                int ker_co = b_id*k_size*k_size + k_size*j + k;
                int im_co = (j+tid_y)*im_size + k + tid_x;
                out[b_id*(im_size - k_size + 1)*(im_size - k_size + 1) + tid_y*(im_size - k_size + 1) + tid_x] += image[im_co] * ker[ker_co];
            }
        }
//     out[b_id*28*28 + tid_y*28 + tid_x] = tanh(out[b_id*28*28 + tid_y*28 + tid_x]);
}

// __global__ void conv(float *image, float *ker, float *out, int k_size, int n_ker){
//     int tid_x = threadIdx.x;
//     int tid_y = threadIdx.y;
//     int b_id = blockIdx.x;
    
//         for(int j = 0; j<5; j++){
//             for(int k = 0; k<5; k++){
//                 int ker_co = b_id*25 + 5*j + k;
//                 int im_co = (j+tid_y)*32 + k + tid_x;
//                 out[b_id*28*28 + tid_y*28 + tid_x] += image[im_co] * ker[ker_co];
//             }
//         }
// //     out[b_id*28*28 + tid_y*28 + tid_x] = tanh(out[b_id*28*28 + tid_y*28 + tid_x]);
// }

__global__ void averagePool(float *in, int len_x, int len_y, int len_z, float *out){
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int b_id = blockIdx.x;
    int surf = len_x*len_y;
    
    out[surf/4*b_id + tid_y*len_x/2 + tid_x] = (in[surf*b_id + tid_y*len_x*2 + tid_x*2] + 
                                              in[surf*b_id + tid_y*len_x*2 + tid_x*2 + 1] + 
                                              in[surf*b_id + tid_y*len_x*2 + len_x + tid_x*2] + 
                                              in[surf*b_id + tid_y*len_x*2 + len_x + tid_x*2 + 1])/4;
}

__device__ float activation_tanh(float M){
    return tanh(M);
}

void init(float *image, int L, int n_ker = 1){
    for(int l = 0; l < n_ker; l++){
        for(int i = 0; i<L; i++){
            for(int j = 0; j<L; j++){
                image[l*L*L + i*L + j] = i;//i + j + 2*l;
            }
        }
    }
}

// identity filter
void init_ker(float *ker, int L){
    ker[(L/2)*L + L/2] = 1;
}

// edge detector horizontal
void init_ker2(float *ker, int L){
    ker[(L*L) + (L/2)*L + L/2 - 1] = -1;
    ker[(L*L) + (L/2)*L + L/2 + 1] = 1;
}

// edge detector vertical
void init_ker3(float *ker, int L){
    ker[2*(L*L) + (L/2 - 1)*L + L/2 ] = -1;
    ker[2*(L*L) + (L/2 + 1)*L + L/2] = 1;
}

void print_vec(float *vec, int L){
    for(int i = 0; i<L; i++){
        printf("%f\n", vec[i]);
    }
}

void print_mat(float *vec, int L, int H, int n_layer = 1){
    for(int l = 0; l<n_layer; l++){
        for(int i = 0; i<H; i++){
            for(int j = 0; j<L; j++)
                printf("%2.2f ", vec[l*L*H + i*L + j]);
            printf("\n");
        }
        printf("\n");
    }
}

void testConv(){
    int L = 32;
    int k_size = 5;
    int n_ker = 3;
    int L_out =  (L - k_size + 1) * (L - k_size + 1) * n_ker;
    int L_ker = k_size * k_size * n_ker;
    
    float *image = (float *) malloc(sizeof(float) * L * L);
    float *ker = (float *) malloc(sizeof(float) * L_ker);
    float *out = (float *) malloc(sizeof(float) * L_out);
    
    float *image_d, *ker_d, *out_d;

    init_ker(ker, k_size);
    init_ker2(ker, k_size);
    init_ker3(ker, k_size);
    
    init(image, L);
    
    cudaMalloc(&image_d, sizeof(float) * L * L);
    cudaMalloc(&ker_d, sizeof(float) * L_ker);
    cudaMalloc(&out_d, sizeof(float) * L_out);
    
    cudaMemcpy(image_d, image, sizeof(float) * L * L, cudaMemcpyHostToDevice);
    cudaMemcpy(ker_d, ker, sizeof(float) * L_ker, cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, sizeof(float) * L_out, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock(28, 28);
    
    conv<<<n_ker, threadPerBlock>>>(image_d, ker_d, out_d, L, k_size, n_ker);
    cudaMemcpy(out, out_d, sizeof(float) * L_out, cudaMemcpyDeviceToHost);
    print_mat(out, 28, n_ker*28);
    cudaDeviceSynchronize();
}

void testMaxPool(){
    int L = 32;
    int n_layer = 2;
    int L_out =  L/2;
    int im_size = L * L * n_layer;
    int out_size = L_out * L_out * n_layer;
    
    float *image = (float *) malloc(sizeof(float) * im_size);
    float *out = (float *) malloc(sizeof(float) * out_size);
    
    float *image_d, *out_d;
    
    init(image, L, n_layer);
    
    cudaMalloc(&image_d, sizeof(float) * im_size);
    cudaMalloc(&out_d, sizeof(float) * out_size);
    
    cudaMemcpy(image_d, image, sizeof(float) * im_size, cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, sizeof(float) * out_size, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock(L_out, L_out);
    
    averagePool<<<n_layer, threadPerBlock>>>(image_d, L, L, n_layer, out_d);
    cudaMemcpy(out, out_d, sizeof(float) * out_size, cudaMemcpyDeviceToHost);
    print_mat(image, L, L, n_layer);
    
    print_mat(out, L_out, L_out, n_layer);
    
    cudaDeviceSynchronize();
}
int main(){
//     testMaxPool();
    testConv();
    
    return 0;
}

