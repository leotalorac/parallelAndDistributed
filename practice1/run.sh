rm results.txt
echo "compiling..."
g++ reduction.cpp -std=c++11 `pkg-config --cflags --libs opencv` -lpthread
echo "runing..."
./a.out "./imgs/img4k.jpg" "./hola.jpg" 1 >> results.txt
./a.out "./imgs/img4k.jpg" "./hola.jpg" 2 >> results.txt
./a.out "./imgs/img4k.jpg" "./hola.jpg" 4 >> results.txt
./a.out "./imgs/img4k.jpg" "./hola.jpg" 8 >> results.txt
./a.out "./imgs/img4k.jpg" "./hola.jpg" 16 >> results.txt
./a.out "./imgs/img1080.jpg" "./hola.jpg" 1 >> results.txt
./a.out "./imgs/img1080.jpg" "./hola.jpg" 2 >> results.txt
./a.out "./imgs/img1080.jpg" "./hola.jpg" 4 >> results.txt
./a.out "./imgs/img1080.jpg" "./hola.jpg" 8 >> results.txt
./a.out "./imgs/img1080.jpg" "./hola.jpg" 16 >> results.txt
./a.out "./imgs/img720.jpg" "./hola.jpg" 1 >> results.txt
./a.out "./imgs/img720.jpg" "./hola.jpg" 2 >> results.txt
./a.out "./imgs/img720.jpg" "./hola.jpg" 4 >> results.txt
./a.out "./imgs/img720.jpg" "./hola.jpg" 8 >> results.txt
./a.out "./imgs/img720.jpg" "./hola.jpg" 16 >> results.txt
