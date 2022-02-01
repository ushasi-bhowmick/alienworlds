import random
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size
import matplotlib.animation as animation
import time
start_time = time.time()

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
Plcoordx = np.array([-1,-0.5,0.5,1,0.5,-0.5])
Plcoordy = np.array([0,-np.sqrt(3)/2,-np.sqrt(3)/2,0,np.sqrt(3)/2,np.sqrt(3)/2])
Rorbit = 12
elevation = 0

#multi abstract objects
Allx = np.array([3*Plcoordx,2*Plcoordx,1*Plcoordx])
Ally =  np.array([3*Plcoordy,2*Plcoordy,1*Plcoordy])
Rorbits=[20,30,40]
ph_offset=np.array([0,np.pi/3, np.pi/2])
o_vel=np.array([2,1,3])

ran_rad=Rstar*np.sqrt(np.random.rand(1000))
ran_th=2*np.pi*np.random.rand(1000)

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

def monte_carlo_multi(polyx, polyy, frame):
    dists=[]
    for i in range(0,len(polyx)):
        if(glp(o_vel[i]*frame+ph_offset)[i]>np.pi/2 and glp(o_vel[i]*frame+ph_offset[i])<3*np.pi/2): 
            dists.append(np.zeros(len(ran_rad)))
        else: 
            distarr=np.asarray([in_or_out(ran_rad[j]*np.cos(ran_th[j]),ran_rad[j]*np.sin(ran_th[j]),polyx[i],polyy[i]) 
                for j in range(len(ran_rad))])
            dists.append(distarr)
    frac = np.sum(np.sum(np.asarray(dists), axis=0)>0)/len(ran_rad)
    return(frac)

def get_transit_lc():
    shape_trajectory=[]
    lc = []
    for frame in thframe:
        tcoordsh = np.transpose(np.asarray([y_rot(np.array([Plcoordx[i],Plcoordy[i],0]), frame) for i in range(len(Plcoordx))]))
        tcoordsh = np.asarray([Rorbit*np.sin(frame)*np.ones(len(Plcoordx))+tcoordsh[0],tcoordsh[1], 
            Rorbit*np.cos(frame)*np.ones(len(Plcoordx))+tcoordsh[2]])

        if(2*(glp(frame)>np.pi/2 and glp(frame)<3*np.pi/2)):
            area = 0
        else: 
            area = monte_carlo(tcoordsh[0],tcoordsh[1])
        shape_trajectory.append(tcoordsh)
        lc.append(1-area)

    return(np.asarray(shape_trajectory), np.asarray(lc))

def get_transit_lc_multi():
    shape_trajectory=[]
    lc = []
    for frame in thframe:
        tcoordlist = []
        for x in range(0,len(Allx)):
            tcoordsh = np.transpose(np.asarray([y_rot(np.array([Allx[x,i],Ally[x,i],0]), frame) for i in range(len(Allx[x]))]))
            tcoordsh = np.asarray([Rorbit*np.sin(o_vel[x]*frame+ph_offset[x])*np.ones(len(Plcoordx))+tcoordsh[0],tcoordsh[1], 
                Rorbit*np.cos(o_vel[x]*frame+ph_offset[x])*np.ones(len(Plcoordx))+tcoordsh[2]])
            tcoordlist.append(tcoordsh)

        #print(np.asarray(tcoordlist).shape)
        tcoordlist = np.array(tcoordlist)
        area = monte_carlo_multi(tcoordlist[:,0],tcoordlist[:,1], frame)
        shape_trajectory.append(tcoordlist)
        lc.append(1-area)

    return(np.asarray(shape_trajectory), np.asarray(lc))

sh_tr=[]
thelc=[] 

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

    if(not thelc[frame]<1): 
        zst = 2
        zpl = 1
    else: 
        zst = 1
        zpl = 4
    ax[0].set_aspect(1)
    ax[0].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax[0].fill(Rstar*np.cos(th), Rstar*np.sin(th), zorder = zst, color='#fff44f')

    ax[0].scatter(ran_rad*np.cos(ran_th), ran_rad*np.sin(ran_th),  marker=',', s=1,color='#ffa343', zorder=3)
    ax[0].set_xlim(-Rorbit*1.2,Rorbit*1.2)
    ax[0].set_ylim(-Rorbit*1.2,Rorbit*1.2)

    #ax[0].fill(sh_tr[frame,0],sh_tr[frame,1], zorder=zpl, color='black', edgecolor='gray')
    for i in range(len(Allx)):
        ax[0].fill(sh_tr[frame,i,0],sh_tr[frame,i,1], zorder=zpl, color='black', edgecolor='gray')
    ax[1].scatter(thframe[frame], thelc[frame], color='red', marker='.')
    return ln,
 
sh_tr, thelc = get_transit_lc_multi()
#np.random.shuffle(ran_rad)
#sh_tr, lc2 = get_transit_lc()
#np.random.shuffle(ran_th)
#sh_tr, lc3 = get_transit_lc()
#thelc = np.mean(np.array([lc1,lc2,lc3]), axis=0)

print("--- %s seconds ---" % (time.time() - start_time))
print("here:", sh_tr.shape, thelc.shape)
ani = animation.FuncAnimation(fig, update, frames=np.arange(0,len(thframe)), interval=1,
                    init_func=init)

#writergif = animation.PillowWriter(fps=20) 
#ani.save('animation_dm_turn_4.gif', writer=writergif)

plt.show() 