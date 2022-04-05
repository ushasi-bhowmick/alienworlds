import random
from turtle import color, pos
from xml.dom.minidom import Element
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from aliensims import Simulator, Megastructure
import time

#just redoing some animation for the presentation...
sim = Simulator(100,100,200, limb_u1=0.9)
meg = Megastructure(500,True,20, ecc=0.6, per_off=np.pi/2)
sim.add_megs(meg)
road, frame, lc = sim.simulate_transit()

plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
fig = plt.figure(figsize=(8,4))
fig.patch.set_facecolor('#101010')
ax = plt.subplot2grid((1,1), (0, 0))
ln, = plt.plot([], [], 'ro')

th = np.linspace(0,2*np.pi, 200)

temp = np.array([[x['x'] for x in el] for el in road.traj])
maxorb = max(np.abs(temp.reshape(-1)))
thframe = np.linspace(-np.pi,np.pi,100)


#print(len(road.traj), road.centres[0][1])

def init():
    ln.set_data([], [])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    props = dict(boxstyle='round', facecolor='black', alpha=0.5, pad=1)
    #txt = "Main Panel: "+str(np.round(2*np.pi*Rorb/(n*Rstar),2))+"$R_{st}$\nOrbit: "+str(Rorb/Rstar)+"$R_{st}$\nu: "+str(0
    #    )+"\ne: "+str(0)

    #ax3.text(0.5, 0.3, txt, fontsize=9,transform=ax3.transAxes,  horizontalalignment='center',
    #        verticalalignment='center', linespacing=2, bbox=props, color='white')

    theta = np.arange(-np.pi, np.pi+np.pi/4, step=(np.pi / 4))
    return ln,


def update(frame):
    ax.clear()
    zst=1
    zpl=[0 if(np.all(el[frame]['z']<0)) else 3 for el in road.traj]
    ax.set_aspect(3)
    plt.axis("equal")
    ax.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax.fill(road.Rstar*np.cos(th), road.Rstar*np.sin(th), zorder = zst, color='#ffa343')
    ax.scatter(road.MCscatter_x, road.MCscatter_y,  marker='.', s=1,color='#fff44f', zorder=2)
    ax.set_xlim(-maxorb*1.2,maxorb*1.2)
    ax.set_ylim(-maxorb*1.2,maxorb*1.2)

    i=0
    cnt = np.array(road.centres[0])
    ax.scatter(cnt[:,0], cnt[:,1],  marker='_', s=1,color='red', zorder=2)
    for el in road.traj:
        ax.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='#8a3324', alpha=1)
        i+=1
    return ln,


ani = animation.FuncAnimation(fig, update, frames=np.arange(0,len(frame)), interval=1,
                    init_func=init)

#plt.tight_layout()
#writergif = animation.PillowWriter(fps=15) 
#ani.save('letsgo4.gif', writer=writergif, savefig_kwargs=dict(facecolor='#101010'))
plt.show()