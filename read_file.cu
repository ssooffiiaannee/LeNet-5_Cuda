#include <stdio.h>
#include <iostream>
#include <vector>
#include "h5cpp"
// #include <stdlib.h>
// #include <unistd.h>

int main(){
//     h5::fd_t fd = h5::open("my.h5", H5F_ACC_RDWR);
//     h5::ds_t ds = h5::open(fd,"model_weights/conv2d/conv2d/bias:0");
    std::vector<float> myvec(10*10, 3);
    auto err = h5::read( fd, "model_weights/conv2d/conv2d/bias:0", myvec.data(), h5::count{10,10}, h5::offset{5,0} );
    
    return 0;
}