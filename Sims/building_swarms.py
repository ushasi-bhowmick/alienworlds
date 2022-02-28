import random
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dy
import time
from multiprocessing import Process, Pool

start_time = time.time()

#redo the animation frames coz this is slightly different...
#we start by randomly placing panels on the dyson sphere

n = 15
Rst = 100
no_pt = 200
Rorb =200
maxout = 10
rinp = 0.9*np.pi*Rorb/(n*np.sqrt(2))
print(rinp)


#random initialization of dyson swarm.
possibilities=[]


for j in range(-int(n/2),int(n/2)+1):
    newn = int(np.pi*Rorb*np.cos(j*np.pi/n)/(rinp*np.sqrt(2)))
    if(j==0): newn=n
    for i in range(0,2*int(newn)):
        possibilities.append([j,i, newn])

np.random.shuffle(possibilities)

print("Total:",len(np.array(possibilities)), np.array(possibilities).shape)

sum_road = []
sum_lc = []

sim1 = dy.Simulator(Rst, 10000, no_pt, np.pi, limb=0.0)

for it in range(0,maxout):
    sim1.megs=[]
    for el in possibilities[:int(2**it)]:
        j=el[0]
        i=el[1]
        newn = el[2]
        meg = dy.Megastructure(Rorb*np.cos(j*np.pi/n), False, isrot=True,elevation=Rorb*np.sin(j*np.pi/n), ph_offset=i*np.pi/newn)
        meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
        meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
        meg.Plcoords=meg.rotate([1,0,0],-j*np.pi/n)
        meg.Plcoords=meg.rotate([0,1,0],i*np.pi/newn)
        sim1.add_megs(meg)
    #possibilities = np.delete(possibilities, np.arange(0,track_ind), axis = 0)
    #print("check total:", len(possibilities))
        
    print("Accepted:",len(sim1.megs))
    road, lc = sim1.simulate_transit()
    sum_road.append(road)
    sum_lc.append(lc)
    sim1.initialize()



print("--- %s seconds ---" % (time.time() - start_time))
print(len(sum_road), len(sum_lc))




#redo the animation frame coz we need a progressive animation showing
#evolution of dyson swarm

plt.style.use('dark_background')
plt.rcParams["font.family"] = "serif"
fig = plt.figure(figsize=(7,7))
fig.patch.set_facecolor('#101010')
ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
xdata, ydata = [], []
ln, = plt.plot([], [], 'ro')
th = np.linspace(0,2*np.pi, 200)

temp = np.array([[x['x'] for x in el] for el in road.traj])
maxorb = max(np.abs(temp.reshape(-1)))
thframe = np.linspace(-np.pi,np.pi,no_pt)

fr_sum = np.tile(np.arange(0,no_pt,3), maxout)



def init():
    ln.set_data([], [])
    ax2.set_xlabel('Phase')
    ax2.set_ylabel('Flux')
    ax2.set_xlim(-np.pi, np.pi)
    ax2.set_ylim(-0.1,1.3)
    ax3.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    plt.suptitle('Dyson Swarm in Construction')
    plt.figtext(0.45, 0.93, "@ushasibhowmick", fontsize=8, color='#F33434')

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    props = dict(boxstyle='round', facecolor='black', alpha=0.5, pad=1)
    txt = "Panel: "+str(np.round(rinp*np.sqrt(2)/Rst,2))+"$R_{st}$\nOrbit: "+str(Rorb/Rst)+"$R_{st}$\nu: "+str(0
        )+"\ne: "+str(0)

    ax3.text(0.5, 0.3, txt, fontsize=9,transform=ax3.transAxes,  horizontalalignment='center',
            verticalalignment='center', linespacing=2, bbox=props, color='white')

    ax2.set_ylabel('Flux')
    ax2.set_xlabel('Phase')
    theta = np.arange(-np.pi, np.pi+np.pi/4, step=(np.pi / 4))
    ax2.set_xticks(theta)
    ax2.set_xticklabels([ '-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])
    return ln,

it = -1
def update(frame):
    global it
    
    if(frame==0): 
        it+=1
        ax2.clear()
        ax2.set_ylim(-0.1,1.1)
        #ax2.grid(which="major",alpha=0.6, color='gray', ls=':')
        ax2.set_ylabel('Flux')
        ax2.set_xlabel('Phase')
        theta = np.arange(-np.pi, np.pi+np.pi/4, step=(np.pi / 4))
        ax2.set_xticks(theta)
        ax2.set_xticklabels([ '-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])
    if(it==maxout): it = 0

    props = dict(boxstyle='round', facecolor='black',edgecolor='#ffa343', pad=1)
    ax3.text(0.5, 0.8, "No. of Panels:\n"+str(len(sum_road[it].traj)), fontsize=9,transform=ax3.transAxes,  horizontalalignment='center',
            verticalalignment='center', linespacing=2, bbox=props, color='#ffa343')

    ax1.clear()
    zst=1
    zpl=[0 if np.all(el[frame]['z']<0) else 3 for el in sum_road[it].traj]
    ax1.set_aspect(1)
    ax1.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
    ax1.fill(sum_road[it].Rstar*np.cos(th), sum_road[it].Rstar*np.sin(th), zorder = zst, color='#ffa343')
    ax1.scatter(road.MCscatter_x, road.MCscatter_y,  marker='.', s=1,color='#fff44f', zorder=2)
    ax1.set_xlim(-maxorb*1.2,maxorb*1.2)
    ax1.set_ylim(-maxorb*1.2,maxorb*1.2)

    i=0
    for el in sum_road[it].traj:
        if(it == 0):
            ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='#0E092C', edgecolor='gray', alpha=0.8)
        elif(i>=len(sum_road[it-1].traj)):
            ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='#0E092C', edgecolor='gray', alpha=0.8)
        else:
            ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='#181818', edgecolor='gray', alpha=0.9)
        i+=1
    ax2.plot(thframe, sum_lc[it], color='#F33434')
    return ln,


#net = np.array([np.array([thframe[i]]+np.array(sum_lc)[:,i]) for i in range(no_pt)])
net =np.transpose(np.insert(np.array(sum_lc),0, thframe, axis=0))
ani = animation.FuncAnimation(fig, update, frames=fr_sum, interval=1,
                    init_func=init)

print(np.array(net).shape)
np.savetxt('sphere_conc_1.csv',net,delimiter=' ', header='phase, panels:1,2,4,8,16...')
writergif = animation.PillowWriter(fps=20) 
ani.save('completed_sphere.gif', writer=writergif, savefig_kwargs=dict(facecolor='#101010'))

#plt.show() 