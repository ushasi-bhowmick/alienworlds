import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import matplotlib.animation as animation
from astropy.io import ascii
import os
import GetLightcurves as gc
import tensorflow as tf


#data=np.loadtxt('training_data/Xtrain_av_raw200_2_2d0.csv',delimiter=',')
#dir = os.listdir('../../processed_directories/sem_seg_op/')
dir2 = os.listdir('../../processed_directories/thebestonesofar/expand_test_result/')

#print(data.shape)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(15,5))
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
    return ln,

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
    return ln,'''

def update3(frame):
    ax.clear()
    ip,tp,pp,sm,ss,plp,fpsp = gc.read_tfr_record('../../processed_directories/thebestonesofar/expand_test_result/'+dir2[frame],
        ['input','true_map','pred_map','scale_median','scale_std','pl_peaks','fps_peaks'],
        ['ar','ar','ar','fl','fl','ar','ar'], 
        [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int16, tf.int16])

    tp = np.reshape(tp, (len(tp),4000,3))
    pp = np.reshape(pp, (len(pp),4000,3))

    totpp = np.concatenate(pp, axis=0)
    tottp = np.concatenate(tp, axis=0)
    totip = np.concatenate(ip, axis=0)

    minima = min(totip)

    xdata=np.arange(0,len(totip),1)
    
    counts=np.asarray([np.argmax([totpp[el,0],totpp[el,1],totpp[el,2]*0.75]) for el in xdata])
    pl=np.where(counts==0)[0]
    fps=np.where(counts==1)[0]
    bkg=np.where(counts==2)[0]

    ax.set_xlabel('time')
    ax.set_ylabel('flux')
    ax.set_title(str(frame)+": "+dir2[frame])
  
    ax.plot(xdata,tottp[:,0]*minima*1.5,marker='_',color='#F93822FF', ls='None')
    ax.plot(xdata,tottp[:,1]*minima*1.5,marker='_',color='#FDD20EFF', ls='None')
    
    ax.plot(xdata[bkg],totip[bkg],color='#CCCCCC',marker='.',ls='None',label='bkg')
    ax.plot(xdata[fps],totip[fps],color='#FDD20EFF',marker='.',ls='None',label='fps') 
    ax.plot(xdata[pl],totip[pl],color='#F93822FF',marker='.',ls='None',label='pl')

    
    #ax.plot(xdata,-dta['PRED_FPS_MAP'],marker='.')
    ax.legend()
    return ln,

#ani = animation.FuncAnimation(fig, update2, frames=np.arange(0,len(data),1), interval=1000,
#                    init_func=init)
ani = animation.FuncAnimation(fig, update3, frames=np.arange(0,int(len(dir2)),1), interval=1000,
                    init_func=init)

#writergif = animation.PillowWriter(fps=1) 
#ani.save('animation3.gif', writer=writergif)

plt.show() 