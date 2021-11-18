import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import matplotlib.animation as animation
import pandas as pd
from astropy.io import ascii
import os


#data=np.loadtxt('training_data/Xtrain_av_raw200_2_2d0.csv',delimiter=',')
dir = os.listdir('../../processed_directories/sem_seg_op/')

#print(data.shape)
plt.style.use('dark_background')
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')

def init():
    ln.set_data([], [])
    ax.set_xlabel('time')
    ax.set_ylabel('flux')
    
    return ln,

'''
def update(frame):
    ax.clear()
    xdata=np.arange(0,len(data[frame]),1)
    ydata=data[frame]
    #print(len(xdata),len(ydata))
    #xdata.append(frame)
    #ydata.append(np.sin(frame))
    ax.plot(xdata,ydata,marker='.')
    med=np.median(data[frame][0:30])
    std=np.std(data[frame][0:30])
    ax.plot((med-0.7*std)*np.ones(len(data[frame])),color='black',label=med/std)
    ax.legend()
    return ln,'''

def update2(frame):
    ax.clear()
    dta = ascii.read('../../processed_directories/sem_seg_op/'+dir[frame])
    xdata=np.arange(0,4000,1)
    ydata=dta['RAW']
    minima = min(ydata)
    print('call')
    
    counts=np.asarray([np.argmax([dta['PRED_PL_MAP'][el],dta['PRED_FPS_MAP'][el],dta['PRED_BKG_MAP'][el]/2]) for el in xdata])
    pl=np.where(counts==0)[0]
    fps=np.where(counts==1)[0]
    bkg=np.where(counts==2)[0]

    ax.set_xlabel('time')
    ax.set_ylabel('flux')
    ax.set_title(str(frame)+": "+dir[frame])
    print(np.asarray(dta['PL_MAP']*minima*1.5).shape)
  
    ax.plot(xdata,dta['PL_MAP']*minima*1.5,marker='_',color='#F93822FF', ls='None')
    ax.plot(xdata,dta['FPS_MAP']*minima*1.5,marker='_',color='#FDD20EFF', ls='None')
    
    ax.plot(xdata[bkg],ydata[bkg],color='#CCCCCC',marker='.',ls='None',label='bkg')
    ax.plot(xdata[fps],ydata[fps],color='#FDD20EFF',marker='.',ls='None',label='fps') 
    ax.plot(xdata[pl],ydata[pl],color='#F93822FF',marker='.',ls='None',label='pl')

    
    #ax.plot(xdata,-dta['PRED_FPS_MAP'],marker='.')
    ax.legend()
    return ln,

#ani = animation.FuncAnimation(fig, update2, frames=np.arange(0,len(data),1), interval=1000,
#                    init_func=init)
ani = animation.FuncAnimation(fig, update2, frames=np.arange(0,int(len(dir)),1), interval=1000,
                    init_func=init)

#writergif = animation.PillowWriter(fps=1) 
#ani.save('animation3.gif', writer=writergif)

plt.show() 