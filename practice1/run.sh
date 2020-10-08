rm results.txt
echo "compiling..."
g++ reduction.cpp -std=c++11 `pkg-config --cflags --libs opencv` -lpthread
echo "runing..."
./a.out "./imgs/img4k.jpg" "./img_result/img4k_1t.jpg" 1 >> results.txt
./a.out "./imgs/img4k.jpg" "./img_result/img4k_2t.jpg" 2 >> results.txt
./a.out "./imgs/img4k.jpg" "./img_result/img4k_4t.jpg" 4 >> results.txt
./a.out "./imgs/img4k.jpg" "./img_result/img4k_8t.jpg" 8 >> results.txt
./a.out "./imgs/img4k.jpg" "./img_result/img4k_16t.jpg" 16 >> results.txt
./a.out "./imgs/img1080.jpg" "./img_result/img1080_1t.jpg" 1 >> results.txt
./a.out "./imgs/img1080.jpg" "./img_result/img1080_2t.jpg" 2 >> results.txt
./a.out "./imgs/img1080.jpg" "./img_result/img1080_4t.jpg" 4 >> results.txt
./a.out "./imgs/img1080.jpg" "./img_result/img1080_8t.jpg" 8 >> results.txt
./a.out "./imgs/img1080.jpg" "../img_result/img1080_16t.jpg" 16 >> results.txt
./a.out "./imgs/img720.jpg" "./img_result/img720_1t.jpg" 1 >> results.txt
./a.out "./imgs/img720.jpg" "./img_result/img720_2t.jpg" 2 >> results.txt
./a.out "./imgs/img720.jpg" "./img_result/img720_4t.jpg" 4 >> results.txt
./a.out "./imgs/img720.jpg" "./img_result/img720_8t.jpg" 8 >> results.txt
./a.out "./imgs/img720.jpg" "./img_result/img720_16t.jpg" 16 >> results.txt
