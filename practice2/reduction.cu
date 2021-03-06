#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define HEIGHT 480
#define WIDTH 720
#define CHANNELS 3
#define ITERATIONS 20

//function to get time
#ifndef TIMER_H
#define TIMER_H

typedef unsigned long long timestamp_t;

static timestamp_t

get_timestamp ()
{
  struct timeval now;
  gettimeofday (&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}
#endif

Mat outImg(HEIGHT, WIDTH, CV_8UC3); 
Mat originalImg;

__global__ void nearest_neighbour_scaling(unsigned char *input_image, unsigned char *output_image,int width_input, int height_input) {
    
    struct dimension
    {
        int height;
        int width;
        float x_ratio;
        float y_ratio;
    } src_dim;


    src_dim.height = height_input;
    src_dim.width = width_input;
    src_dim.y_ratio = ((float)src_dim.height/HEIGHT);
    src_dim.x_ratio = ((float)src_dim.width/WIDTH);


	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    int row,col = 0;
    int b;
    const int input_width_step = width_input * CHANNELS;
    const int output_width_step = WIDTH * CHANNELS;

    if ((xIndex < WIDTH) && (yIndex < HEIGHT)){
        row = ceil(yIndex *  src_dim.y_ratio);
        col = ceil(xIndex * src_dim.x_ratio);
        for (int channel = 0; channel < CHANNELS; channel++){
            *(output_image + (yIndex * output_width_step + xIndex * CHANNELS + channel)) =  *(input_image + (row * input_width_step + col * CHANNELS + channel));
        }
    }
}



int main(int argc, char* argv[]) {
    
    /*Parameter inputs: Source, Out and number of threads*/
    const string src = argv[1];
    const string dst = argv[2];
    const int threads = atoi(argv[3]);

    originalImg = imread(src);
   

    /*Allocate Space*/
    const int size_input = sizeof(unsigned char) *  originalImg.size().height *  originalImg.size().width * CHANNELS; 
    const int size_output = sizeof(unsigned char) * WIDTH * HEIGHT * CHANNELS;
    unsigned char *input_image_pointer, *output_image_pointer;

    cudaMalloc<unsigned char>(&input_image_pointer, size_input);
    cudaMalloc<unsigned char>(&output_image_pointer, size_output);
    cudaMemcpy(input_image_pointer, originalImg.ptr(), size_input, cudaMemcpyHostToDevice);

    /*Time Recording*/
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start, NULL);

   
    const dim3 threadsPerBlock(threads, threads);
    const dim3 numBlocks(WIDTH / threadsPerBlock.x, HEIGHT / threadsPerBlock.y);
    for(int i = 0; i < ITERATIONS; i++){
        nearest_neighbour_scaling<<<numBlocks, threadsPerBlock>>>(input_image_pointer, output_image_pointer, originalImg.size().width, originalImg.size().height);
    }
   

    /*Stop Time recording*/
    cudaEventRecord(end, NULL);
    cudaEventSynchronize(end);
    
    /*Calculate Time*/
    float time = 0.0f;
    cudaEventElapsedTime(&time, start, end);
    float totalTime = time / (ITERATIONS * 1.0f);
    printf("%.8f",totalTime);
  
    cudaMemcpy(outImg.ptr(), output_image_pointer, size_output, cudaMemcpyDeviceToHost);

    imwrite(dst, outImg);

    cudaFree(input_image_pointer);
    cudaFree(output_image_pointer);

    return 0;
}
