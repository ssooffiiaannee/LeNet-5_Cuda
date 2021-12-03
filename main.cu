#include <stdio.h>
//  token : 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

class LeNet5{
    public:
      LeNet5(){

      }
};

int main(){
    int devCount;
    cudaGetDeviceCount(&devCount);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("%d\n", devProp.maxThreadsPerBlock);
    
    return 0;
}