echo "compiling..."
g++ reduction.cpp -std=c++11 `pkg-config --cflags --libs opencv` -lpthread
echo "runing..."
./a.out "./imgs/img720.jpg" "./hola.jpg" 1