#include <stdio.h>
#include <math.h>
#include "weights.h"
#define N 3
void funct(int n){
    printf("%f\n", conv2d_w[0]);
    printf("%f\n", exp(1));
    printf("%d\n", n);
}
#define WIDTH 32
#define HEIGHT 32
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
          if((i < 2) || (i > 29) || (j < 2) || (j > 29) ){
              im[(int) i*WIDTH + j] = 0;
              continue;
          }
          fread(&val, sizeof(unsigned char), 1, fptr);  
          im[(int) i*WIDTH + j]=(double) val/255;
        }
     }
}