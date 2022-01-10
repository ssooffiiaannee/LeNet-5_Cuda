#include <stdio.h>
#include "weights.h"

#define WIDTH 32
#define HEIGHT 32

// nvcc compilation not working properly when jupyter-lab kernel is running

__global__ void Conv2d(double *image, double *ker, double *out, double *biases, int im_size, int k_size, int n_ker, int n_c){    
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
    out[b_id*out_size*out_size + tid_y*(out_size) + tid_x] = tanh(out[b_id*out_size*out_size + tid_y*(out_size) + tid_x] + biases[b_id]);

}


__global__ void averagePool(double *in, int len_x, int len_y, int len_z, double *out){
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int b_id = blockIdx.x;
    int surf = len_x*len_y;
    
    out[surf/4*b_id + tid_y*len_x/2 + tid_x] = (in[surf*b_id + tid_y*len_x*2 + tid_x*2] + 
                                              in[surf*b_id + tid_y*len_x*2 + tid_x*2 + 1] + 
                                              in[surf*b_id + tid_y*len_x*2 + len_x + tid_x*2] + 
                                              in[surf*b_id + tid_y*len_x*2 + len_x + tid_x*2 + 1])/4;
}

__device__ double activation_tanh(double M){
    return tanh(M);
}

__device__ double activation_softmax(double *vec, int n, int i){
    double sum = 0;
    
    for(int j = 0; j<n; j++){
        sum += exp(vec[j]);
    }
    
    return exp(vec[i])/sum;
}

void init(double *image, int L, int n_ker = 1){
    for(int l = 0; l < n_ker; l++){
        for(int i = 0; i<L; i++){
            for(int j = 0; j<L; j++){
                image[l*L*L + i*L + j] = i;//i + j + 2*l;
            }
        }
    }
}


void print_vec(double *vec, int L){
    for(int i = 0; i<L; i++){
        printf("%f\n", vec[i]);
    }
}

void print_mat(double *vec, int L, int H, int n_c){
    for(int l = 0; l<n_c; l++){
        for(int i = 0; i<H; i++){
            for(int j = 0; j<L; j++)
//                 printf("%d ", 7*((int) ceil(vec[l*L*H + i*L + j])));
                printf("%1.5f ", vec[l*L*H + i*L + j]);
            printf("\n");
        }
        printf("\n");
    }
}


void Conv1(double *im, double *ker, double *biases, int n_ker, int im_size, int n_c){
    int k_size = 5;
    int k_vol = k_size * k_size * n_ker;
    int out_vol = (WIDTH-k_size + 1)*(HEIGHT-k_size + 1)*n_ker;
    int img_vol = im_size * im_size * n_c;
    
//     double *kernel = (double *) malloc(sizeof(double) k_vol);
    double *out = (double *) malloc(sizeof(double) * out_vol);
                                     
    double *im_d, *out_d, *ker_d, *biases_d;
    cudaMalloc(&im_d, sizeof(double) * img_vol);
    cudaMalloc(&out_d, sizeof(double) * out_vol);
    cudaMalloc(&ker_d, sizeof(double) * k_vol);
    cudaMalloc(&biases_d, sizeof(double) * n_ker);
    
    cudaMemcpy(im_d, im, sizeof(double) * img_vol, cudaMemcpyHostToDevice);
    cudaMemcpy(ker_d, ker, sizeof(double) * k_vol, cudaMemcpyHostToDevice);
    cudaMemcpy(biases_d, biases, sizeof(double) * n_ker, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock((WIDTH-k_size + 1), (HEIGHT-k_size + 1));
    dim3 blockPerGrid(n_ker);
    printf("this==\n");
//     Conv2d<<<blockPerGrid, threadPerBlock>>>();
    Conv2d<<<blockPerGrid, threadPerBlock>>>(im_d, ker_d, out_d, biases_d, im_size, k_size, n_ker, n_c);
    cudaDeviceSynchronize();
    // ### MeanAvPool
    double *outMAP_d;
    int vol_out_MAP = (WIDTH-k_size + 1)*(HEIGHT-k_size + 1) * n_ker;
    cudaMalloc(&outMAP_d, sizeof(double) * img_vol);
    averagePool(out_d, (WIDTH-k_size + 1), (HEIGHT-k_size + 1), n_ker, double *out);
    // ###
    cudaMemcpy(out, out_d, sizeof(double) * out_vol, cudaMemcpyDeviceToHost);
    
    print_mat(out, (WIDTH-k_size + 1), (HEIGHT-k_size + 1), n_ker);
    cudaFree(im_d);
    cudaFree(out_d);
    cudaFree(ker_d);
    cudaFree(biases_d);
    
    free(out);
}

int main(){
    double im[WIDTH * HEIGHT];
    unsigned int magic, nbImg, nbRows, nbCols;
    unsigned char val;
    FILE *fptr;

    //Open File
    if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
        printf("Can't open file");
        exit(1);
    }
    
    fread(&magic, sizeof(int), 1, fptr);
    fread(&nbImg, sizeof(int), 1, fptr);
    fread(&nbRows, sizeof(int), 1, fptr);
    fread(&nbCols, sizeof(int), 1, fptr);
    
    for(int i=0; i<HEIGHT; i++){
        for(int j=0; j<WIDTH; j++){
            if((i < 2) || (i > 29) || (j < 2) || (j > 29) ){ // works
//             if((i < 2) || (i > 29) || (j < 2) || (j > 29) ){ // doesn't work
              im[(int) i*WIDTH + j] = 0;
              continue;
          }
          fread(&val, sizeof(unsigned char), 1, fptr);  
          im[(int) i*WIDTH + j] = (double) val/255;
        }
    }

    Conv1(im, conv2d_w, conv2d_b, 6, 32, 1);
    
    return 0;
}

