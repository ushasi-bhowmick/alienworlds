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

#single planets... start with triangle
Rstar = 10
#Plcoordx = np.array([-4,4,4])
#Plcoordy = np.array([0,4,-4])
#Plcoordx = 3*np.array([-1,-0.5,0.5,1,0.5,-0.5])
#Plcoordy = 3*np.array([0,-np.sqrt(3)/2,-np.sqrt(3)/2,0,np.sqrt(3)/2,np.sqrt(3)/2])
plth = np.linspace(-np.pi,np.pi, 50)
Plcoordx = 3*np.cos(plth)
Plcoordy = 3*np.sin(plth)

Rorbit = 12
elevation = 0

ran_rad=Rstar*np.sqrt(np.random.rand(5000))
ran_th=2*np.pi*np.random.rand(5000)

#check if a point lies inside or outside a polygon
def in_or_out(refx,refy,shx,shy):
    #step 1: eliminate stuff outside the bounding box
    if(refx<min(shx) or refx>max(shx) or refy>max(shy) or refy<min(shy)): return(0)
    #step 2: ray tracing horizontal
    shyt = np.append(shy,shy[0])
    shxt = np.append(shx,shx[0])
    intsecty = (np.asarray([(shyt[i]-refy)*(shyt[i+1]-refy) if(shxt[i]>refx) else 0 
        for i in range(0,len(shyt)-1)])<0).sum()
    #print(len(shyt), intsecty)
    #intsectx = (np.asarray([(shxt[i]-refx)*(shxt[i]-refx) if(shyt[i]>refy) 
    #    else 0 for i in range(0,len(shxt)-1)])<0).sum()
    if(intsecty%2 !=0): return(1)
    else: return(0)

#add a rotation about y axis:
def y_rot(arr, L):
    mat = np.array([[np.cos(L),0,np.sin(L)],[0,1,0],[-np.sin(L),0,np.cos(L)]])
    return(np.matmul(mat,arr))

def glp(el): 
    rem = np.floor(el*0.5/np.pi)
    return(el - rem*2*np.pi)


def monte_carlo(polyx, polyy):
    distarr=np.asarray([in_or_out(ran_rad[i]*np.cos(ran_th[i]),ran_rad[i]*np.sin(ran_th[i]),polyx,polyy) 
        for i in range(len(ran_rad))])
    frac = (distarr).sum()/len(ran_rad)
    return(frac)




def init():
    ln.set_data([], [])
    ax[1].set_xlabel('Phase')
    ax[1].set_ylabel('Flux')
    ax[1].set_xlim(min(thframe),max(thframe))
    ax[1].set_ylim(0.8,1.1)
    plt.suptitle('Transit Simulations')
    plt.figtext(0.8, 0.75, "Star Rad: "+str(Rstar), fontsize=9)
    #plt.figtext(0.8, 0.73, "Planet Rad: "+str(Rpl), fontsize=9)
    plt.figtext(0.8, 0.71, "Orbit: "+str(Rorbit), fontsize=9)
    #plt.figtext(0.8, 0.69, "Offset(pi): "+np.array_str(ph_offset/np.pi, precision=2, suppress_small=True), fontsize=9)
    #plt.figtext(0.8, 0.67, "Velocity: "+str(o_vel), fontsize=9)
    plt.figtext(0.45, 0.9, "@ushasibhowmick", fontsize=7)

    return ln,


def update(frame):
    ax[0].clear()

    tcoordsh = np.transpose(np.asarray([y_rot(np.array([Plcoordx[i],Plcoordy[i],0]), frame) for i in range(len(Plcoordx))]))
    tcoordsh = np.asarray([Rorbit*np.sin(frame)*np.ones(len(Plcoordx))+tcoordsh[0],tcoordsh[1], 
        Rorbit*np.cos(frame)*np.ones(len(Plcoordx))+tcoordsh[2]])

    if(2*(glp(frame)>np.pi/2 and glp(frame)<3*np.pi/2)): 
    #if(np.all(tcoordsh[:,2]<0)): 
        zst = 2
        zpl = 1
        area = 0
    else: 
        zst = 1
        zpl = 4
        #area = monte_carlo(Rorbit*np.sin(frame)*np.ones(len(Plcoordx))+Plcoordx,Plcoordy)
        area = monte_carlo(tcoordsh[0],tcoordsh[1])
    ax[0].set_aspect(1)
    #ax[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax[0].fill(Rstar*np.cos(th), Rstar*np.sin(th), zorder = zst, color='#fff44f')

    ax[0].scatter(ran_rad*np.cos(ran_th), ran_rad*np.sin(ran_th),  marker=',', s=1,color='#ffa343', zorder=3)
    ax[0].set_xlim(-Rorbit*1.2,Rorbit*1.2)
    ax[0].set_ylim(-Rorbit*1.2,Rorbit*1.2)

    ax[0].fill(tcoordsh[0],tcoordsh[1], zorder=zpl, color='black', edgecolor='gray')
    ax[1].scatter(frame, 1-area, color='red', marker='.')
    return ln,

ani = animation.FuncAnimation(fig, update, frames=thframe, interval=1,
                    init_func=init)

#writergif = animation.PillowWriter(fps=20) 
#ani.save('animation_dm_turn_2.gif', writer=writergif)

plt.show() 