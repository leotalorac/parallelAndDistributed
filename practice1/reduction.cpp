#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

#define HEIGHT 480
#define WIDTH 720
#define CHANNELS 3
int threads;

Mat originalImg;
Mat outImg(HEIGHT, WIDTH, CV_8UC3, Scalar(255, 255, 255));

struct dimension
{
    int height;
    int width;
    float x_ratio;
    float y_ratio;
} src_dim;

void nndownscale(void *id_thread)
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


    // Calculate image
    for (int i = 0; i < HEIGHT; i++)
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
}

//imgsource imgout n_threads
int main(int argc, char *argv[])
{
    //Get the source image path, out path and number of threads
    string src_img = argv[1];
    string out_img = argv[2];
    threads = atoi(argv[3]);

    int *ptr;
    nndownscale(ptr);
    imwrite(out_img, outImg);
    return 0;
}