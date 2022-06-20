import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root_scalar
import copy
import time
import pandas as pd
from gfg import in_or_out

import aliensims as dysim

sim = dysim.Simulator(50, 2000, 5000, np.pi, limb_u1=0, limb_u2=0)
meg_mc = dysim.Megastructure(np.sqrt(83.86*50), True, 0.0035*30*5, ecc=0, o_vel=8.82, ph_offset=0.7)
meg_vs = dysim.Megastructure(np.sqrt(154.82*50), True, 0.0086*30*5, ecc=0, o_vel=6.44, ph_offset=0.6)
meg_e = dysim.Megastructure(np.sqrt(215.032*50), True, 0.0091*30*5, ecc=0, o_vel=5.48, ph_offset=0.5)
meg_ms = dysim.Megastructure(np.sqrt(326.84*50), True, 0.005*30*5, ecc=0, o_vel=4.42, ph_offset=0.4)
meg_jp = dysim.Megastructure(np.sqrt(1118.17*50), True, 0.102*30*5, ecc=0, o_vel=2.41, ph_offset=0.3)
meg_st = dysim.Megastructure(np.sqrt(2051.4*50), True, 0.086*30*5, ecc=0, o_vel=1.78, ph_offset=0.2)
meg_ur = dysim.Megastructure(np.sqrt(4128.61*50), True, 0.036*30*5, ecc=0, o_vel=1.25, ph_offset=0.1)
meg_np = dysim.Megastructure(np.sqrt(6450.9*50), True, 0.035*30*5, ecc=0, o_vel=1)
sim.add_megs(meg_mc)
sim.add_megs(meg_vs)
sim.add_megs(meg_e)
sim.add_megs(meg_ms)
sim.add_megs(meg_jp)
sim.add_megs(meg_st)
sim.add_megs(meg_ur)
sim.add_megs(meg_np)

    
road, phase, lc = sim.simulate_transit()
df = pd.read_csv('solarsim2.csv')
lc=df['flux']

plt.style.use('seaborn-bright')
plt.rcParams["font.family"] = "serif"
fig = plt.figure(figsize=(7,7))
fig.patch.set_facecolor('#cccccc')
ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=3)
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)

xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')
th = np.linspace(0,2*np.pi, 200)

temp = np.array([[x['x'] for x in el] for el in road.traj])
maxorb = max(np.abs(temp.reshape(-1)))
thframe = np.linspace(-np.pi,np.pi,len(lc))


def init():
    ln.set_data([], [])
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Flux')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(min(lc)*0.9,1.01)
    plt.suptitle('Solar System in Transit')
    plt.figtext(0.45, 0.93, "@ushasibhowmick", fontsize=8, color='#F33434')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax2.set_ylabel('Flux')
    ax2.set_xlabel('Phase')
    theta = np.arange(-np.pi, np.pi+np.pi/4, step=(np.pi / 4))
    ax2.set_xticks(theta)
    ax2.set_xticklabels([ '-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])
    return ln,

it = -1
def update(frame):
    ax1.clear()
    zst=1
    zpl=[0 if np.all(el[frame]['z']<0) else 2 for el in road.traj]
    ax1.set_aspect(1)
    ax1.fill(road.Rstar*np.cos(th), road.Rstar*np.sin(th), zorder = zst, color='#ffae42')

    ax1.set_xlim(-maxorb*1.01,maxorb*1.01)
    ax1.set_ylim(-maxorb*0.5,maxorb*0.5)
    ax1.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    i=0
    for el in road.traj:
        ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='black', edgecolor='gray')
        i+=1
    ax2.clear()
    ax2.plot(phase[frame-500:frame+500], lc[frame-500:frame+500], color='red')  
    ax2.set_ylim(min(lc[frame-500:frame+500])*0.995,1)
    ax2.set_ylabel('Flux')
    ax2.set_xlabel('Phase')
    #ax2.scatter(phase[frame], lc[frame], color='red', marker='.')
    return ln,


#net = np.array([np.array([thframe[i]]+np.array(sum_lc)[:,i]) for i in range(no_pt)])

ani = animation.FuncAnimation(fig, update, frames=range(500,len(phase)-500,5), interval=1,
                    init_func=init)

# print(np.array(net).shape)
# np.savetxt('sphere_conc_org_1.csv',net,delimiter=' ', header='phase, panels:1,2,4,8,16...')
writergif = animation.PillowWriter(fps=30) 
ani.save('solarsystem2.gif', writer=writergif, savefig_kwargs=dict(facecolor='#cccccc'))
plt.show()