import matplotlib
import matplotlib.pyplot as plt
import numpy as np
threads = [1,2,4,8,16]
f = open("times.txt", "r")
times = []
speed = [1]
sp1=0
for t in range(len(threads)):
    tim = []
    for j in range(10):
        tim.append(float(f.readline()))
    avgtime=np.average(tim)
    times.append(avgtime)
    if(t>0):
        speed.append(sp1/avgtime)
    else:
        sp1 = avgtime
fig, ax = plt.subplots()
ax.plot(threads, times)
ax.set(xlabel='Threads', ylabel='Time (s)',
    title='Threads vs time calculus pi')
ax.grid()
fig.savefig("time.png")
plt.show()
fig, ax = plt.subplots()
ax.plot(threads, speed)
ax.set(xlabel='Threads', ylabel='SpeedUp (s)',
    title='Threads vs speedup calculus pi')
ax.grid()
fig.savefig("speed.png")
plt.show()
