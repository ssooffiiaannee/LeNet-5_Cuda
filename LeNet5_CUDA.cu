#include <stdio.h>
#include "weights.h"
#include "printImg.h"

// nvcc compilation not working when jupyter-lab kernel is running

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
                                                in[surf*b_id + tid_y*len_x*2 + tid_x*2 + len_x] + 
                                                in[surf*b_id + tid_y*len_x*2 + tid_x*2 + len_x + 1])/4;
}

void softmax(double *vec, double *out, int n){
    double sum = 0;
    
    for(int j = 0; j<n; j++){
        sum += exp(vec[j]);
    }
    for(int j = 0; j<n; j++)
        out[j] = exp(vec[j])/sum;
}

void init(double *image, int L, int n_ker){
    for(int l = 0; l < n_ker; l++){
        for(int i = 0; i<L; i++){
            for(int j = 0; j<L; j++){
                image[l*L*L + i*L + j] = i*L + j;//i;//i + j + 2*l;
            }
        }
    }
}


void print_vec(double *vec, int L){
    for(int i = 0; i<L; i++){
        printf("label %d with probability : %f\n", i, vec[i]);
    }
}

void print_mat(double *vec, int L, int H, int n_c){
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

/*
    Apply a 2D Convolution
    @param:
        im : input features
        W, H, n_c : Height, Width and numbre of channels (number of filters of previous layer or 3 for RGB input)
        ker : pointer to kernel weights
        biases : pointer to n_ker biases values.
        n_ker : number of filters to apply
    @return: 
        Pointer to double, pointing to Conv result
*/
double * Conv1(double *im, int H, int W, double *ker, double *biases, int n_ker, int n_c){
    int k_size = 5;
    int k_vol = k_size * k_size * n_ker * n_c;
    int out_vol = (W-k_size + 1)*(H-k_size + 1)*n_ker;
    int img_vol = H * W * n_c;
    
    double *out = (double *) malloc(sizeof(double) * out_vol);
                                     
    double *im_d, *out_d, *ker_d, *biases_d;
    cudaMalloc(&im_d, sizeof(double) * img_vol);
    cudaMalloc(&out_d, sizeof(double) * out_vol);
    cudaMalloc(&ker_d, sizeof(double) * k_vol);
    cudaMalloc(&biases_d, sizeof(double) * n_ker);
    
    cudaMemcpy(im_d, im, sizeof(double) * img_vol, cudaMemcpyHostToDevice);
    cudaMemcpy(ker_d, ker, sizeof(double) * k_vol, cudaMemcpyHostToDevice);
    cudaMemcpy(biases_d, biases, sizeof(double) * n_ker, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock((W-k_size + 1), (H-k_size + 1));
    dim3 blockPerGrid(n_ker);
    printf("################ Conv1 ########################\n\n");
//     Conv2d<<<blockPerGrid, threadPerBlock>>>();
    Conv2d<<<blockPerGrid, threadPerBlock>>>(im_d, ker_d, out_d, biases_d, H, k_size, n_ker, n_c);
    cudaDeviceSynchronize();

    cudaMemcpy(out, out_d, sizeof(double) * out_vol, cudaMemcpyDeviceToHost);
    
    cudaFree(im_d);
    cudaFree(out_d);
    cudaFree(ker_d);
    cudaFree(biases_d);
    
    return out;
}
/*
    Apply a 2D average pooling
    @param:
        in : input features
        W, H, n_c : Height, Width and numbre of channels (number of filters of previous layer)
    @return: 
        Pointer to double, pointing to average pool result
*/
double * averagePool2D(double *in, unsigned int W, unsigned int H, int n_c){
    double *outMAP_d,  *in_d;
    int vol_out_MAP = W * H * n_c / 4;
    int in_vol = W * H * n_c;
    
    double *outMAP = (double *) malloc(sizeof(double) * vol_out_MAP);
    cudaMalloc(&outMAP_d, sizeof(double) * vol_out_MAP);
    cudaMalloc(&in_d, sizeof(double) * in_vol);
    cudaMemcpy(in_d, in, sizeof(double) * in_vol, cudaMemcpyHostToDevice);

    averagePool<<<n_c, { W/2, H/2}>>>(in_d, W, H, n_c, outMAP_d);
    cudaDeviceSynchronize();
    cudaMemcpy(outMAP, outMAP_d, sizeof(double) * vol_out_MAP, cudaMemcpyDeviceToHost);
    
    printf("################ averagePool2D ########################\n\n");

    cudaFree(outMAP_d);
    
    return outMAP;
}

void zero_padding(double *im, int H, int pad, double *out){
    for(int i = 0; i<H + 2*pad; i++){
        for(int j = 0; j<H + 2*pad; j++){
            if((i < pad) || (j < pad) || (i >= H + pad) || (j >= H + pad)){
                out[i*(H + 2*pad) + j] = 0;
                continue;
            }
            out[i*(H + 2*pad) + j] = im[(i - pad)*H + j - pad];
        }
    }
}

/*
    The reshape will reorder the vector of features.
    @param:
        in : input features
        W, H, n_c : Height, Width and numbre of channels (number of filters of previous layer)
        out : pointer to the output vector
    @param: 
*/
__global__ void reshape(double *in, double *out, int H, int W, int n_c){
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = blockIdx.z;
    
    out[i*W*n_c + j*n_c + k] = in[k*(W * H) + i*H + j];
}

double * Flatten(double *in, int H, int W, int n_c){
    double *in_d, *out_d;
    double *out = (double *) malloc(sizeof(double) * H * W * n_c);
    
    cudaMalloc(&in_d, sizeof(double) * W * H * n_c);
    cudaMalloc(&out_d, sizeof(double) * W * H * n_c);
    
    cudaMemcpy(in_d, in, sizeof(double) * W * H * n_c, cudaMemcpyHostToDevice);
    
    dim3 threadPerBlock(1);
    dim3 blockPerGrid(H, W, n_c);
    reshape<<<blockPerGrid, threadPerBlock>>>(in_d, out_d, H, W, n_c);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out, out_d, sizeof(double) * W * H * n_c, cudaMemcpyDeviceToHost);
    
    cudaFree(out_d);
    
    return out;
}

/*
    Matrix multipliaction for Weights x Dense output
    @param:
        M1, M2 : M1 is H x W matrix representing the weights, M2 is H2 x 1 matrix for flatten and dense layers output
        W, H, H2 : shapes of weights matrix and dense dense outputs (H2 number of units of dense layer)
        act : Integer, choosing activation function, 1 for tanh activation function, 0 for softmax
        Mout : pointer to the output vector.
    @param: 
*/
__global__ void cudaMatrixMult(double *M1, double *M2, double *b, double *Mout, int H, int W, int H2, int act){
    int l = blockIdx.y;
    int c = blockIdx.x;

    for(int i = 0; i<W; i++){
        Mout[l*H2 + c] += M1[l*W + i]*M2[c*H2 + i];
    }
    if(act == 1)
        Mout[l*H2 + c] = tanh(Mout[l*H2 + c]  + b[l*H2 + c]);
    else
        Mout[l*H2 + c] = Mout[l*H2 + c]  + b[l*H2 + c];
}

void zero_init(double *vec, int sz){
    for(int i = 0; i<sz; i++) 
        vec[i] = 0;
}
/*
    Dense layer
    @param:
        in : input features
        w, b : Height, Width and numbre of channels (number of filters of previous layer)
        in_sz : number of units of the previous layer.
        out_sz : number of units in the dense layer.
        activation : activation type, 1 for tanh, 0 for softmax
    @param: 
        Pointer output results, vector of size out_sz
*/
double * Dense(double *in, double *w, double *b, int in_sz, int out_sz, int activation){
    double *b_d, *w_d, *in_d, *out_d;
    double *out = (double *) malloc(sizeof(double) * out_sz);
    zero_init(out, out_sz);
    
    cudaMalloc(&in_d, sizeof(double) * in_sz);
    cudaMalloc(&w_d, sizeof(double) * out_sz * in_sz);
    cudaMalloc(&b_d, sizeof(double) * out_sz);
    cudaMalloc(&out_d, sizeof(double) * out_sz);
    
    cudaMemcpy(in_d, in, sizeof(double) * in_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(w_d, w, sizeof(double) * out_sz * in_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, sizeof(double) * out_sz, cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out, sizeof(double) * out_sz, cudaMemcpyHostToDevice);
    
    printf("################ Dense ########################\n\n");
    dim3 n_blocks(1, out_sz);
    cudaMatrixMult<<<n_blocks, 1>>>(w_d, in_d, b_d, out_d, out_sz, in_sz, 1, activation);
    cudaDeviceSynchronize();
    cudaMemcpy(out, out_d, sizeof(double) * out_sz, cudaMemcpyDeviceToHost);
    
    return out;
}
/*
    Returns index of the largest value in the vector.
    @param:
        vec : input vector
        n : size of the vector
    @return: index of the largest value in the vector
*/
int argmax(double *vec, int n){
    int id = 0;
    double mn = -1;
    for(int i = 0; i<n; i++){
        if(mn < vec[i]){
            mn = vec[i];
            id = i;
        }
    }
    return id;
}

int main(){
    int image_number;
    int H = 28, pad = 2;
    int H_p = H + 2*pad;
    double im[H * H];
    double im_p[H_p * H_p];
    
    printf("Enter a number between 0 and 59999 : ");
    scanf("%d", &image_number);
    if(image_number < 0 || image_number >59999){
        printf("!!! Number has to be between 0 and 59999 !!!\n\n");
        printf("Enter a number between 0 and 59999 : ");
        scanf("%d", &image_number);
    }
    
    // #############################
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
    for(int a = 0; a<image_number + 1; a++){
        for(int i=0; i<H; i++){
            for(int j=0; j<H; j++){
              fread(&val, sizeof(unsigned char), 1, fptr);  
              im[(int) i*H + j] = (double) val/255;
            }
        }
    }
    imgColorPrint(H, H, im);
    
    // ################ Model Starts Here ###############
    
    zero_padding(im, H, pad, im_p);
    
//     print_mat(im_p, H_p, H_p , 1);
    
    double *l1;
    l1 = Conv1(im_p, H_p, H_p, conv2d_w, conv2d_b, 6, 1);
    l1 = averagePool2D(l1, H, H, 6);
//     print_mat(l1, H/2, H/2, 6);
    
    l1 = Conv1(l1, H/2, H/2, conv2d_1_w, conv2d_1_b, 16, 6);
//     print_mat(l1, H/2 - 4, H/2 - 4, 16);
    
    l1 = averagePool2D(l1, H/2 - 4, H/2 - 4, 16);
//     print_mat(l1, (H/2 - 4)/2, (H/2 - 4)/2, 16);
    
    l1 = Flatten(l1, (H/2 - 4)/2, (H/2 - 4)/2, 16);
//     print_vec(l1, (H/2 - 4)/2 * (H/2 - 4)/2 * 16 );
    int units_1 = 120, units_2 = 84, units_3 = 10;
//     printf("%f\n", l1[83]);
    l1 = Dense(l1, dense_w, dense_b, (H/2 - 4)/2 * (H/2 - 4)/2 * 16, units_1, 1);
    l1 = Dense(l1, dense_1_w, dense_1_b, units_1, units_2, 1);
    l1 = Dense(l1, dense_2_w, dense_2_b, units_2, units_3, 0);
    
    printf("############### Predictions ############\n");
    double out[units_3];
    softmax(l1, out, units_3);
    print_vec(out, units_3);
    printf("\n################# Answer ###################\n\n");
    int pred_lab = argmax(out, units_3);
    printf("Model predicted %i with probability %f.\n", pred_lab, out[pred_lab]);
    printf("\n#############################################\n");
    
    free(l1);
    return 0;
}

