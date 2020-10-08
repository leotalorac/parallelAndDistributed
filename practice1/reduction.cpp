#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define HEIGHT 480
#define WIDTH 720
#define CHANNELS 3
#define ITERATIONS 10

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


Mat originalImg;
Mat outImg(HEIGHT, WIDTH, CV_8UC3);
int threads; 
struct dimension
{
    int height;
    int width;
    float x_ratio;
    float y_ratio;
} src_dim;

void *nndownscale(void *arg)
{
    // define the dimention and downsize of the image
    src_dim.height = originalImg.size().height;
    src_dim.width = originalImg.size().width;
    src_dim.y_ratio = ((float)src_dim.height/HEIGHT);
    src_dim.x_ratio = ((float)src_dim.width/WIDTH);
    // starting settings
    int row = 0;
    int col = 0;

    int id = *(int *)arg;
    int n = outImg.size().height / threads;
    int inicio = n * id;
    int fin = inicio + n;

    // Calculate image
    for (int i = inicio; i < fin; i++)
    {
        for(int j =0; j<outImg.size().width;j++){
            row = ceil(i * src_dim.y_ratio);
            col = ceil(j * src_dim.x_ratio);
            for(int c =0 ; c<CHANNELS;c++){
				outImg.at<uchar>(i,j * CHANNELS + c) = originalImg.at<uchar>(row,col * CHANNELS + c); 
            }
        }
    }
    return 0;
}

//imgsource imgout n_threads
int main(int argc, char* argv[]) {    
    //Get the source image path, out path and number of threads
    string src_img = argv[1];
    string out_img = argv[2];
    threads = atoi(argv[3]);
    originalImg = imread(src_img);

    int threadId[threads], i, *retval;
    pthread_t thread[threads];
    timestamp_t start, end;
    double avg;

    start = get_timestamp();
	timestamp_t startUnity, endUnity;
    double avgUnity;


	printf("Path Imagen: %s \n", src_img.c_str());
	printf("Numero de Hilos:%d\n", threads);
    for(int k = 0; k<ITERATIONS; k++){
		startUnity = get_timestamp();
    	for(i = 0; i < threads; i++){
            threadId[i] = i;
            pthread_create(&thread[i], NULL, nndownscale, &threadId[i]);
        }

        for(i = 0; i < threads; i++){
            pthread_join(thread[i], (void **)&retval);
        }
		endUnity = get_timestamp();
        avgUnity = (endUnity - startUnity);

		printf("%f\n",avgUnity/(double)1000);
     }

    end = get_timestamp();
    avg = (end - start)/(double)ITERATIONS;

    printf("%f\n", avg/(double)1000);
    imwrite(out_img, outImg);
    return 0;
}