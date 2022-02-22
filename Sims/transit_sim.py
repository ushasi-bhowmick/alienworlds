import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import matplotlib.animation as animation
from scipy.optimize import root_scalar

plt.style.use('dark_background')
fig, ax = plt.subplots(2,1,figsize=(8,10))
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')
th = np.linspace(0, 2*np.pi, 200)

thframe = np.pi*np.linspace(-1, 1, 300)

#single planets...
Rstar = 10
Rpl = 1
Rorbit = 15
elevation = 0

#multiple systems...
Rpls = [2,5,3]
Rorbits=[20,30,40]
ph_offset=np.array([0,np.pi/3, np.pi/2])
o_vel=np.array([2,1,3])
#ph_offset = np.array([0,0,0])

#ran_rad=Rstar*np.sqrt(np.random.rand(1000))
ran_th=2*np.pi*np.random.rand(6000)
u=0.9
ran_rad=Rstar*np.power(np.random.rand(6000),(1/3))

def Prob(x, z, u):
    k=1-u/2
    y = ((1-u)*x + u*x**2/2)/k -z
    return(y)

temp_z =np.random.rand(6000)
ran_x = []
for el in temp_z:
    try: sol = root_scalar(Prob,args=(el,u),bracket=[0,np.pi/2])
    except: continue
    ran_x.append(sol.root)

ran_x = np.sin(np.arccos(2*np.array(ran_x)-1))

def glp(el): 
    rem = np.floor(el*0.5/np.pi)
    return(el - rem*2*np.pi)


def monte_carlo(centrex, centrey,no_pt):
    distarr = np.sqrt( (centrex-ran_rad*ran_x*np.sin(ran_th))**2+ (centrey-ran_rad*ran_x*np.cos(ran_th))**2)
    frac = (distarr<Rpl).sum()/len(ran_x)
    return(frac)

def monte_carlo_multi(centrex, centrey,no_pt, frame):
    dists=[]
    for i in range(0,len(centrex)):
        if(glp(o_vel[i]*frame+ph_offset)[i]>np.pi/2 and glp(o_vel[i]*frame+ph_offset[i])<3*np.pi/2): 
            dists.append(np.zeros(no_pt))
        else: 
            distarr = np.sqrt((centrex[i]-ran_rad*np.cos(ran_th))**2+ (centrey[i]-ran_rad*np.sin(ran_th))**2) 
            dists.append(distarr<Rpls[i])
    frac = np.sum(np.sum(np.asarray(dists), axis=0)>0)/no_pt
    return(frac)

def init():
    ln.set_data([], [])
    ax[1].set_xlabel('Phase')
    ax[1].set_ylabel('Flux')
    ax[1].set_xlim(min(thframe),max(thframe))
    ax[1].set_ylim(0.9,1.01)
    plt.suptitle('Transit Simulations')
    plt.figtext(0.8, 0.75, "Star Rad: "+str(Rstar), fontsize=9)
    plt.figtext(0.8, 0.73, "Planet Rad: "+str(Rpls), fontsize=9)
    plt.figtext(0.8, 0.71, "Orbit: "+str(Rorbits), fontsize=9)
    plt.figtext(0.8, 0.69, "Offset(pi): "+np.array_str(ph_offset/np.pi, precision=2, suppress_small=True), fontsize=9)
    plt.figtext(0.8, 0.67, "Velocity: "+str(o_vel), fontsize=9)
    plt.figtext(0.45, 0.9, "@ushasibhowmick", fontsize=7)

    return ln,


def update(frame):
    ax[0].clear()
    
    if(2*(glp(frame)>np.pi/2 and glp(frame)<3*np.pi/2)): 
        zst = 2
        zpl = 1
        area = 0
    else: 
        zst = 1
        zpl = 4
        area = monte_carlo(Rorbit*np.sin(frame),elevation,1000)
    ax[0].set_aspect(1)
    ax[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax[0].fill(Rstar*np.cos(th), Rstar*np.sin(th), zorder = zst, color='#fff44f')
    ax[0].scatter(ran_rad*ran_x*np.sin(ran_th), ran_rad*ran_x*np.cos(ran_th),  marker=',', s=1,color='#ffa343', zorder=3)
    ax[0].set_xlim(-Rorbit*1.2,Rorbit*1.2)
    ax[0].set_ylim(-Rorbit*1.2,Rorbit*1.2)
    ax[0].fill(Rorbit*np.sin(frame)+Rpl*np.cos(th), elevation+Rpl*np.sin(th), zorder=zpl, color='black', edgecolor='gray')
    ax[1].scatter(frame, 1-area, color='red', marker='.')
    return ln,

def update_multi(frame):
    ax[0].clear()
    zst = 1
    zpls=np.asarray([3*(glp(o_vel[i]*frame+ph_offset[i])<np.pi/2 or glp(o_vel[i]*frame+ph_offset[i])>3*np.pi/2) 
        for i in range(len(ph_offset))])
    #print(zpls)

    area = monte_carlo_multi(Rorbits*np.sin(o_vel*frame+ph_offset),np.zeros(len(ph_offset)),len(ran_rad), frame)
    ax[0].set_aspect(1)
    ax[0].set_xlim(-max(Rorbits)*1.2, max(Rorbits)*1.2)
    ax[0].set_ylim(-max(Rorbits)*1.2,max(Rorbits)*1.2)

    ax[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax[0].scatter(ran_rad*np.cos(ran_th), ran_rad*np.sin(ran_th),  marker=',', s=1,color='#ffa343', zorder=2)
    ax[0].fill(Rstar*np.cos(th), Rstar*np.sin(th), zorder = zst, color='#fff44f')
    for i in range(len(Rpls)):
        ax[0].fill(Rorbits[i]*np.sin(o_vel[i]*frame+ph_offset[i])+Rpls[i]*np.cos(th), Rpls[i]*np.sin(th), zorder=zpls[i], color='black', edgecolor='gray')
    
    ax[1].scatter(frame, 1-area, color='red', marker='.')
    return ln,
   

    
    


ani = animation.FuncAnimation(fig, update, frames=thframe, interval=1,
                    init_func=init)

#writergif = animation.PillowWriter(fps=20) 
#ani.save('animation_mpl_3.gif', writer=writergif)

plt.show() 