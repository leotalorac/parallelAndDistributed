#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define HEIGHT 480
#define WIDTH 720
#define CHANNELS 3
#define ITERATIONS 10

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

//Objects that store the images
Mat originalImg;
Mat outImg(HEIGHT, WIDTH, CV_8UC3);

//Number of Threads
int threads; 

//Structure to store src image data
struct dimension
{
    int height;
    int width;
    float x_ratio;
    float y_ratio;
} src_dim;

//Algorithm
void *nndownscale(void *arg)
{
    // define the dimention and downsize of the image
    src_dim.height = originalImg.size().height;
    src_dim.width = originalImg.size().width;
    src_dim.y_ratio = ((float)src_dim.height/HEIGHT);
    src_dim.x_ratio = ((float)src_dim.width/WIDTH);

    // starting settings
    int row,col = 0;

    // Calculate the rows of the image that this thread will create
    int id = *(int *)arg;
    int n = HEIGHT/ threads;
    int start = n * id;
    int end = start + n;

    // Calculate image
    for (int i = start; i < end; i++)
    {
        for(int j =0; j<WIDTH;j++){
            //Will take the pixel x-distant located in row, ol from one another depending on the y and x ratio
            row = ceil(i * src_dim.y_ratio);
            col = ceil(j * src_dim.x_ratio);

            //Will take the 3 colors that make that pixel
            for(int c =0 ; c<CHANNELS;c++){
                //and assigned them to our result image
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

    //src image
    originalImg = imread(src_img);

    //Thread variables
    int threadId[threads], i, *retval;
    pthread_t thread[threads];

    //Time Variables
    timestamp_t start, end, startCycle, endCycle;
    double avg, avgCycle;

    start = get_timestamp();


	printf("Path Image: %s \n", src_img.c_str());
	printf("Number of Threads:%d\n", threads);

    //10 Iterations per Thread
    for(int k = 0; k<ITERATIONS; k++){

        //Get start time
		startCycle = get_timestamp();

        //Create Threads and call the algorithm passing the id of thread as a parameter
    	for(i = 0; i < threads; i++){
            threadId[i] = i;
            pthread_create(&thread[i], NULL, nndownscale, &threadId[i]);
        }

        //Join Threads
        for(i = 0; i < threads; i++){
            pthread_join(thread[i], (void **)&retval);
        }

        //Get end time and calculate average
		endCycle = get_timestamp();
        avgCycle = (endCycle - startCycle);

		printf("%f\n",avgCycle/(double)1000);
     }

    //Get end time and calculate average
    end = get_timestamp();
    avg = (end - start)/(double)ITERATIONS;

    printf("%f\n", avg/(double)1000);

    //Writes result image in specified path
    imwrite(out_img, outImg);
    return 0;
}