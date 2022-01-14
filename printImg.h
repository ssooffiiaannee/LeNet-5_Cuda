#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


void charBckgrndPrint(char *str, int rgb){
//   printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
    printf("\033[48;2;%d;%d;%dm", rgb, 0, 0);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, double *img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,(int) 255*img[row*width + col]);
    }
    printf("\n");
  }
}

