// #include <stdio.h>
// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define RESULT_WIDTH 720
#define RESULT_HEIGHT 480
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

Mat outImg(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3); 
Mat originalImg;

__global__ void nearest_neighbour_scaling(
    unsigned char *originalImg, 
    unsigned char *outImg,
    int width_input, 
    int height_input,
    int channels_input,
    int width_output, 
    int height_output,
    int channels_output) {
    const float x_ratio = (width_input + 0.0) / width_output;
    const float y_ratio = (height_input + 0.0) / height_output;

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    int px = 0, py = 0; 
    const int input_width_step = width_input * channels_input;
    const int output_width_step = width_output * channels_output;

    if ((xIndex < width_output) && (yIndex < height_output)){
        py = ceil(yIndex * y_ratio);
        px = ceil(xIndex * x_ratio);
        for (int channel = 0; channel < CHANNELS; channel++){
            *(outImg + (yIndex * output_width_step + xIndex * channels_output + channel)) =  *(originalImg + (py * input_width_step + px * channels_input + channel));
        }
    }
}



int main(int argc, char* argv[]) {
    
    //Get the source image path, out path and number of thread
    const string src = argv[1];
    const string dst = argv[2];
    const int threads = atoi(argv[3]);

    //src image
    originalImg = imread(src);

    cudaEvent_t start, end;
    
    const int size = sizeof(unsigned char) * originalImg.cols * originalImg.rows * CHANNELS;
    printf("%n",outImg.cols );
    const int output_bytes = sizeof(unsigned char) * outImg.cols * outImg.rows * CHANNELS; 

    unsigned char *d_input, *d_output;
    cudaMalloc<unsigned char>(&d_input, size);
    cudaMalloc<unsigned char>(&d_output, output_bytes);

    cudaMemcpy(d_input, originalImg.ptr(), size, cudaMemcpyHostToDevice);
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    
    int width_input = originalImg.cols;
    int height_input = originalImg.rows;
    int channels_input = originalImg.channels();
    int width_output = outImg.cols;
    int height_output = outImg.rows;
    int channels_output = outImg.channels();

    cudaEventRecord(start, NULL);
    const dim3 threadsPerBlock(threads, threads);
    const dim3 numBlocks(width_output / threadsPerBlock.x, height_output / threadsPerBlock.y);
    for(int i = 0; i < ITERATIONS; i++){
            nearest_neighbour_scaling<<<numBlocks, threadsPerBlock>>>(d_input, d_output, width_input, height_input, CHANNELS, width_output, height_output, CHANNELS);
    }
    cudaEventRecord(end, NULL);
    cudaEventSynchronize(end);
  
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, end);
    float secPerMatrixMul = msecTotal / (ITERATIONS * 1.0f);
    printf("%.8f",secPerMatrixMul);
  
    cudaMemcpy(outImg.ptr(), d_output, output_bytes, cudaMemcpyDeviceToHost);

    imwrite(dst, outImg);

    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
