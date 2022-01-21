import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import matplotlib.animation as animation


plt.style.use('dark_background')
fig, ax = plt.subplots(2,1,figsize=(8,10))
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')
th = np.linspace(0, 2*np.pi, 200)

thframe = np.pi*np.linspace(-1, 1, 300)

Rstar = 10
Rpl = 5
Rorbit = 20
elevation = 0

#multiple systems...
Rpls = [2,3,5]
Rorbits=[20,30,40]
ph_offset=np.array([0,np.pi/3, np.pi/2])

zst = 1
zpl = 2

def monte_carlo(centrex, centrey,no_pt):
    ran_rad=np.random.uniform(0, Rstar, no_pt)
    ran_th=np.random.uniform(0, 2*np.pi, no_pt)
    distarr = np.sqrt( (centrex-ran_rad*np.cos(ran_th))**2+ (centrey-ran_rad*np.sin(ran_th))**2)
    frac = (distarr<Rpl).sum()/no_pt
    return(frac)

def monte_carlo_multi(centrex, centrey,no_pt, frame):
    ran_rad=np.random.uniform(0, Rstar, no_pt)
    ran_th=np.random.uniform(0, 2*np.pi, no_pt)
    fracs=[]
    for i in range(0,len(centrex)):
        if(frame+ph_offset[i]>0 and frame+ph_offset[i]<np.pi): continue
        distarr = np.sqrt((centrex[i]-ran_rad*np.cos(ran_th))**2+ (centrey[i]-ran_rad*np.sin(ran_th))**2) 
        fracs.append((distarr<Rpls[i]).sum())
    frac = np.asarray(fracs).sum()/no_pt
    return(frac)

def init():
    ln.set_data([], [])
    ax[1].set_xlabel('Phase')
    ax[1].set_ylabel('Flux')
    ax[1].set_xlim(-np.pi,np.pi)
    ax[1].set_ylim(-0.1,1.1)
    plt.suptitle('Transit Simulations')
    plt.figtext(0.8, 0.75, "Star Rad: "+str(Rstar), fontsize=10)
    plt.figtext(0.8, 0.7, "Planet Rad: "+str(Rpls), fontsize=10)
    plt.figtext(0.8, 0.65, "Orbit: "+str(Rorbits), fontsize=10)

    return ln,


def update(frame):
    ax[0].clear()
    
    if(frame>0): 
        zst = 2
        zpl = 1
        area = 0
    else: 
        zst = 1
        zpl = 2
        area = monte_carlo(Rorbit*np.cos(frame),elevation,100)
    ax[0].set_aspect(1)
    ax[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax[0].fill(Rstar*np.cos(th), Rstar*np.sin(th), zorder = zst, color='#fff44f')
    ax[0].set_xlim(-Rorbit*1.2,Rorbit*1.2)
    ax[0].set_ylim(-Rorbit*1.2,Rorbit*1.2)
    ax[0].fill(Rorbit*np.cos(frame)+Rpl*np.cos(th), elevation+Rpl*np.sin(th), zorder=zpl, color='black', edgecolor='gray')
    ax[1].scatter(frame, 1-area, color='red', marker='.')
    return ln,

def update_multi(frame):
    ax[0].clear()
    zst = 1
    zpls=np.asarray([2*((frame+el)<0 or (frame+el)>np.pi) for el in ph_offset])
    #print(zpls)

    area = monte_carlo_multi(Rorbit*np.cos(frame+ph_offset),np.zeros(3),100, frame)
    ax[0].set_aspect(1)
    ax[0].set_xlim(-max(Rorbits)*1.2, max(Rorbits)*1.2)
    ax[0].set_ylim(-max(Rorbits)*1.2,max(Rorbits)*1.2)

    ax[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax[0].fill(Rstar*np.cos(th), Rstar*np.sin(th), zorder = zst, color='#fff44f')
    for i in range(len(Rpls)):
        ax[0].fill(Rorbits[i]*np.cos(frame+ph_offset[i])+Rpls[i]*np.cos(th), Rpls[i]*np.sin(th), zorder=zpls[i], color='black', edgecolor='gray')
    
    ax[1].scatter(frame, 1-area, color='red', marker='.')
    return ln,
   

    
    


ani = animation.FuncAnimation(fig, update_multi, frames=thframe, interval=1,
                    init_func=init)

writergif = animation.PillowWriter(fps=20) 
ani.save('animation_pl3.gif', writer=writergif)

#plt.show() 