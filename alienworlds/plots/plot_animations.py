import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import matplotlib.animation as animation
import pandas as pd
import os


data=np.loadtxt('training_data/Xtrain_av_stitch.csv',delimiter=',')
print(data.shape)
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ln.set_data([], [])
    #ax.set_xlim(0, 2*np.pi)
    #ax.set_ylim(-1, 1)
    return ln,

def update(frame):
    ax.clear()
    xdata=np.arange(0,len(data[frame]),1)
    ydata=data[frame]
    #print(len(xdata),len(ydata))
    #xdata.append(frame)
    #ydata.append(np.sin(frame))
    ax.plot(xdata,ydata,marker='.')
    med=np.median(data[frame])
    std=np.std(data[frame])
    ax.plot((med-0.7*std)*np.ones(len(data[frame])),color='black')
    return ln,

ani = animation.FuncAnimation(fig, update, frames=np.arange(0,len(data),1), interval=500,
                    init_func=init)
plt.show() 
