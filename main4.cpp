#include <stdio.h>
#include "printImg.h"

#define WIDTH 28
#define HEIGHT 28

int main() {
  int i, j;
  int color[3]={255,0,0};
  unsigned int magic, nbImg, nbRows, nbCols;
  unsigned char val;
  FILE *fptr;

  // Malloc image
   int * img = (int *) malloc(HEIGHT * WIDTH*sizeof(int));

  //Open File
  if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }

  //Read File
  fread(&magic, sizeof(int), 1, fptr);
  fread(&nbImg, sizeof(int), 1, fptr);
  fread(&nbRows, sizeof(int), 1, fptr);
  fread(&nbCols, sizeof(int), 1, fptr);

    for(int N = 0; N < 8; N++){
      for(i=0; i<HEIGHT; i++){
        for(j=0; j<WIDTH; j++){ 
          fread(&val, sizeof(unsigned char), 1, fptr);  
          img[i*WIDTH + j] = (int)val*color[0]/255;
        }
      }
         imgColorPrint(HEIGHT, WIDTH, img);
    }
  
  exit(EXIT_SUCCESS);
}