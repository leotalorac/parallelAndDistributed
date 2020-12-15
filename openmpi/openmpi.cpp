#include "mpi.h"
#include <opencv2/opencv.hpp>
#include <bits/stdc++.h>

using namespace cv;
using namespace std;

#define RESULT_WIDTH 720
#define RESULT_HEIGHT 480
#define ITERATIONS 10
#define MS 1000000.0
#define MAXTASKS 32

typedef unsigned long long timestamp_t;

// Create result image of 720x480 pixels with 3 channels
Mat result_image(RESULT_HEIGHT, RESULT_WIDTH, CV_8UC3, Scalar(255, 255, 255)); 
Mat img;

/**
Implementation of Nearest Neighbour interpolation algorithm to down 
sample the source image
*/
void nearest_neighbour_scaling(int id, int tasks) {
    const int channels_source = im  g.channels(), channels_target = result_image.channels(); // Número de canales (3)

    const int width_source = img.size().width;
    const int height_source = img.size().height;
    const int width_target = result_image.size().width;
    const int height_target = result_image.size().height;

    const float x_ratio = (width_source + 0.0) / width_target;
    const float y_ratio = (height_source + 0.0) / height_target;
    int px = 0, py = 0; 
    
    uchar *ptr_source = nullptr;
    uchar *ptr_target = nullptr;

    int n_rows = height_target / tasks;
    int initial_y = n_rows * id;
    int end_y = initial_y + n_rows;

    for (; initial_y < end_y; initial_y++) {
        ptr_target = result_image.ptr<uchar>(initial_y);
        // Iterate over the cols
        for (int j = 0; j < width_target; j++) {
            py = ceil(initial_y * y_ratio);
            px = ceil(j * x_ratio);
            ptr_source = img.ptr<uchar>(py);
            
            // Calculate the value of the i,j pixel for each channel
            for (int channel = 0; channel < channels_target; channel++){
                ptr_target[j * channels_target + channel] =  ptr_source[channels_source * px + channel];
            }
        }
    }
}

int main(int argc, char* argv[]) { 
    if (argc != 4) {
        cout << "Arguments are not complete. Usage: image_path image_result_path algorithm" << endl;
        return 1;
    }
    // Read parameters 1- source path, 2- Destination path
    string source_image_path = argv[1];
    string result_image_path = argv[2];

    int tasks, iam, root=0;
    int total_pixels = RESULT_WIDTH * RESULT_HEIGHT * 3;
    double start, end, abs_time;
    double max_time = 0.0, min_time = -1.0, avg_time = 0.0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &iam);
    
    int pixels_per_proc = total_pixels / tasks;

    // Read the image from the given source path
    img = imread(source_image_path);
    if(img.empty()) {
        cout << "The image " << source_image_path << " was not found\n";
        return 1;
    }

    start = MPI_Wtime();
    nearest_neighbour_scaling(iam, tasks);
    unsigned char *ptr_target = (result_image.ptr() + pixels_per_proc * iam);
    MPI_Gather(ptr_target, pixels_per_proc, MPI_UNSIGNED_CHAR, 
                result_image.ptr(), pixels_per_proc, MPI_UNSIGNED_CHAR, 
                root, MPI_COMM_WORLD);

    end = MPI_Wtime();
    abs_time = end - start;
    MPI_Reduce(&end, &max_time, 1, MPI_DOUBLE, MPI_MAX, root, MPI_COMM_WORLD);

    MPI_Reduce(&start, &min_time, 1, MPI_DOUBLE, MPI_MIN, root, MPI_COMM_WORLD);

    MPI_Reduce(&abs_time, &avg_time, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    
    if (iam == root){
        avg_time /= tasks;
        printf("Min: %f, Max: %f, Diff: %f, Avg: %f\n", min_time, max_time, max_time - min_time, avg_time);
        imwrite(result_image_path, result_image); //Write the image to a file
    }
    MPI_Finalize();
}

// mpic++ image_scaling_openmpi.cpp -o image_scaling_openmpi `pkg-config --cflags --libs opencv` -lm

// mpirun -np 4 ./image_scaling_openmpi ./images/image1_1080p.jpg ./images/image1_480p.jpg Nearest