import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# open data
f = open("results.txt", "r")
# 1 2 4 8 16 threads
# 720p 1080p 4k
lines = f.readlines()
titles = ["4k","1080","720"]
t = [1,2,4,8,12,16]
onethreadtime = []
readindex = 0
speedups = []
times =[]
for i in range(3): 
    avg = []
    speedup = [0]
    for j in range(6):
        readindex+=12
        avgtem =float(lines[readindex])
        avg.append(avgtem)
        readindex+=1
        if(j ==0):
            onethreadtime.append(avgtem)
        else:
            speedup.append(onethreadtime[i]/avgtem)
    speedups.append(speedup)
    times.append(avg)
    # graphics time
    fig, ax = plt.subplots()
    ax.plot(t, avg)
    ax.set(xlabel='threads', ylabel='time (ms)',
        title='time of ' + titles[i])
    ax.grid()
    fig.savefig("./graphics/timesvsthreads-"+titles[i]+".pdf")
    # graphics speed
    fig, ax = plt.subplots()
    ax.plot(t, speedup)
    ax.set(xlabel='threads', ylabel='speed up',
        title='speedup of ' + titles[i])
    ax.grid()
    fig.savefig("./graphics/speedvsthreads-"+titles[i]+".pdf")
# graphics speed
fig, ax = plt.subplots()
ax.plot(t, speedups[0],'r-',label="4k")
ax.plot(t,speedups[1],'b-',label="1080p")
ax.plot(t,speedups[2],'g-',label="720p")
ax.legend()
ax.set(xlabel='threads', ylabel='speed up',
    title='speedup of three resolutions OpenMP')
ax.grid()
fig.savefig("./graphics/speedvsthreads.pdf")
# graphics times
fig, ax = plt.subplots()
ax.plot(t, times[0],'r-',label="4k")
ax.plot(t,times[1],'b-',label="1080p")
ax.plot(t,times[2],'g-',label="720p")
ax.legend()
ax.set(xlabel='threads', ylabel='time (ms)',
    title='times of three resolutions OpenMP')
ax.grid()
fig.savefig("./graphics/timesvsthreads.pdf")
