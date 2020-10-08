echo "compiling..."
g++ reduction.cpp -std=c++11 `pkg-config --cflags --libs opencv` -lpthread
echo "runing..."
./a.out "./imgs/image1_1080p.jpg" "./hola.jpg" 1