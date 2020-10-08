echo "compiling..."
g++ reduction.cpp -std=c++11 `pkg-config --cflags --libs opencv` -lpthread
echo "runing..."
./a.out "./imgs/img4k.jpg" "./hola.jpg" 1 
./a.out "./imgs/img4k.jpg" "./hola.jpg" 2 
./a.out "./imgs/img4k.jpg" "./hola.jpg" 4 
./a.out "./imgs/img4k.jpg" "./hola.jpg" 8 
./a.out "./imgs/img4k.jpg" "./hola.jpg" 16 
./a.out "./imgs/img1080.jpg" "./hola.jpg" 1 
./a.out "./imgs/img1080.jpg" "./hola.jpg" 2 
./a.out "./imgs/img1080.jpg" "./hola.jpg" 4 
./a.out "./imgs/img1080.jpg" "./hola.jpg" 8 
./a.out "./imgs/img1080.jpg" "./hola.jpg" 16 
./a.out "./imgs/img720.jpg" "./hola.jpg" 1 
./a.out "./imgs/img720.jpg" "./hola.jpg" 2 
./a.out "./imgs/img720.jpg" "./hola.jpg" 4 
./a.out "./imgs/img720.jpg" "./hola.jpg" 8 
./a.out "./imgs/img720.jpg" "./hola.jpg" 16 
