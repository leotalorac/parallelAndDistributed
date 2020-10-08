import numpy
import matplotlib
# open data
f = open("results.txt", "r")
# 1 2 4 8 16 threads
# 720p 1080p 4k
for i in range(1):
    for j in range(1):
        for k in range(10):
            f.read()
        avg = float(f.read())
        print(avg)
