#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>
#include <sys/time.h>

using namespace cv;
using namespace std;

#define HEIGHT 480
#define WIDTH 720
#define CHANNELS 3

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
Mat outImg(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));
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
    src_dim.y_ratio = ((float)src_dim.height / HEIGHT);
    src_dim.x_ratio = ((float)src_dim.width / WIDTH);
    // starting settings
    int row = 0;
    int col = 0;
    uchar *source_row_pointer = nullptr;
    uchar *target_row_pointer = nullptr;


    int id = *(int *)arg;
    int n_raws = outImg.size().height / threads;
    int initial_y = n_raws * id;
    int end_y = initial_y + n_raws;

    // Calculate image
    for (int i = initial_y; i < end_y; i++)
    {
        target_row_pointer = outImg.ptr<uchar>(i);
        for (int j = 0; j < outImg.size().width; j++)
        {
            row = ceil(i * src_dim.y_ratio);
            col = ceil(j * src_dim.x_ratio);
            source_row_pointer = originalImg.ptr<uchar>(row);
            for (int c = 0; c < CHANNELS; c++)
            {
                target_row_pointer[j * CHANNELS + c] = source_row_pointer[col * CHANNELS + c];
            }
        }
    }
    return 0;
}

//imgsource imgout n_threads
int main(int argc, char *argv[])
{
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
    for(i = 0; i < threads; i++){
            threadId[i] = i;
            pthread_create(&thread[i], NULL, nndownscale, &threadId[i]);
        }

        for(i = 0; i < threads; i++){
            pthread_join(thread[i], (void **)&retval);
        }
    end = get_timestamp();
    avg = (end - start);

    printf("Time Elapsed: %f\n", avg/(double)1000);
    imwrite(out_img, outImg);
    return 0;
}