#include <stdio.h>

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


int main(){
//     testMaxPool();
    testConv();
    
    return 0;
}

