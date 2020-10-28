import matplotlib
import matplotlib.pyplot as plt
import numpy as np

f = open("datacuda.txt", "r")
threads = [128,256,512,1024]
blocks =[2,4,8,16,32]

for i in range(4):
    times =[]
    for j in range(4):
        times.append(float(f.readline()))
    fig, ax = plt.subplots()
    ax.plot(threads, times)
    ax.set(xlabel='Threads', ylabel='Time (s)',
        title='Threads vs time #blocks: ' + str(blocks[i]))
    ax.grid()
    fig.savefig("./graphicscuda/time-"+str(blocks[i])+".png")
    # plt.show()
for i in range(4):
    times =[]
    for j in range(4):
        times.append(float(f.readline()))
    fig, ax = plt.subplots()
    ax.plot(threads, times)
    ax.set(xlabel='Threads', ylabel='Speed Up',
        title='Threads vs speedup #blocks: ' + str(blocks[i]))
    ax.grid()
    fig.savefig("./graphicscuda/speedup-"+str(blocks[i])+".png")
    # plt.show()