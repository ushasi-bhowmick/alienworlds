import random
#from turtle import color, pos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dy
import time
from multiprocessing import Process, Pool

start_time = time.time()

#redo the animation frames coz this is slightly different...
#we start by randomly placing panels on the dyson sphere
n = 60
phi = 2*np.pi/n

Rstar = 100
Rorb =200
a = 2*np.pi*Rorb/n
no_pt = 200
maxout = 11


#random initialization of dyson swarm: we make a stash of possibilities... means an array that tells the sum total of all
#the positions that can be occupied in the swarm. The first axis goes in y direction, while the second axis goes in x direction


def list_of_places():
    possibilities=[]
    for j in range(-int(n/4),int(n/4)+1):
        for i in range(0,n):
            possibilities.append([i,j])

    np.random.shuffle(possibilities)
    # print("Total:",len(np.array(possibilities)), np.array(possibilities).shape)
    # print(possibilities)
    possibilities = np.array(possibilities)
    return(possibilities)



#now we must iterate over the possibilies and select as many as we need, until we run out of elements.
def one_arbitrary_shape(prev,how_many,possibilities):
    xmax = max(possibilities[:,0])
    xmin = min(possibilities[:,0])
    ymin = min(possibilities[:,1])
    ymax = max(possibilities[:,1])

    #delete all the existing blocks... by index, so that we have the list of reamining possibilities
    new=list(np.copy(prev))
    ind = np.array([np.where((possibilities == np.array(el)).all(axis=1))[0] for el in prev]).reshape(-1)
    possibilities_left=np.delete(possibilities,ind, axis=0)
    
    x=-1
    tab=0
    while(tab<how_many):
        x+=1
        if(len(possibilities_left)==0): break
        #condition where the sphere is complete
        if(how_many>=len(possibilities_left)): return(possibilities)
        if(x>=len(possibilities_left)): x=0

        #pick one out of the possibilities left and check if it lies adjacent to the point existing shape
        temp=possibilities_left[x]
        if(np.any((np.array(new)==temp+np.array([1,0])).all(axis=1))): new.append(temp)
        elif(np.any((np.array(new)==temp+np.array([-1,0])).all(axis=1))): new.append(temp)
        elif(np.any((np.array(new)==temp+np.array([0,1])).all(axis=1))): new.append(temp)
        elif(np.any((np.array(new)==temp+np.array([0,-1])).all(axis=1))): new.append(temp)
        else: continue
        #delete the newly picked one
        tab+=1
        possibilities_left=np.delete(possibilities_left,x, axis=0)

    #Sprint('new:', np.array(new))
    return(new)

#plot of the shape
def plot_shape(shape):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    ax.set_aspect('equal', adjustable='box')
    th = np.linspace(0,2*np.pi,200)
    print(phi)
    ax.fill(Rstar*np.cos(th), Rstar*np.sin(th), color='yellow',zorder=1)
    for el in shape:
        i=el[0]
        j=el[1]
        ang1 = j*phi
        ang2 = j*phi
        if(j*phi+phi/2 > np.pi/2): ang1 = np.pi/2 - phi/2
        elif(j*phi-phi/2 < -np.pi/2): ang2 = np.pi/2 + phi/2
        a1=2*np.pi*Rorb*np.cos(ang1+phi/2)/n
        a2=2*np.pi*Rorb*np.cos(ang2-phi/2)/n
        a=2*np.pi*Rorb*np.cos(j*phi)/n
        coords=np.array([[a1/2,a/2,0],[-a1/2,a/2,0],[-a2/2,-a/2,0],[a2/2,-a/2,0]])
        meg = dy.Megastructure(Rorb*np.cos(j*phi), False, isrot=True,elevation=Rorb*np.sin(j*phi), Plcoords=coords, ph_offset=i*phi)
        meg.Plcoords=meg.rotate([1,0,0],-j*phi)
        meg.Plcoords=meg.rotate([0,1,0],i*phi)
        meg.Plcoords,C = meg.translate(0)
        print(meg.Plcoords[:,0],meg.Plcoords[:,2])
        if(np.any(meg.Plcoords[:,2]<0)):
            ax.fill(meg.Plcoords[:,0],meg.Plcoords[:,1],color='#000000', edgecolor='gray', alpha=0.8, zorder=0)
        else:
            ax.fill(meg.Plcoords[:,0],meg.Plcoords[:,1],color='#000000', edgecolor='gray', alpha=0.8, zorder=2)
    plt.show()
        
#now we need to do something about the degeneracies.
def translate_or_flip(shape, nextsh, possibilities):
    xmax = max(possibilities[:,0])
    xmin = min(possibilities[:,0])
    ymin = min(possibilities[:,1])
    ymax = max(possibilities[:,1])
    print(xmax, ymax, xmin, ymin)

    shapemaxx=max(shape[:,0])
    shapeminx=min(shape[:,1])

    #identical from different seeds
    if((shape==nextsh).all()): return(1)

    #translate along x
    for i in range(0,shapemaxx-shapeminx,1):
        newshape=shape+[i,0]
        if(np.any(newshape[:,0]>xmax)):
            newshape[np.where(newshape>xmax)[0],0] = newshape[np.where(newshape>xmax)[0],0] - xmax
        if(np.any(newshape[:,0]<xmin)):
            newshape[np.where(newshape<xmin)[0],0] = newshape[np.where(newshape>xmax)[0],0] + xmax
        if((newshape==nextsh).all()): return(1)

    #flip along y
    newshape=np.array([[el[0], -el[1]] for el in shape])
    print(shape, newshape)
    if((newshape==nextsh).all()): return(1) 
    else: return 0



#function to run the simulation for an arbitrary shape
def lc_of_shape(stash):
    sim1 = dy.Simulator(Rstar, 5000, no_pt, np.pi, limb_u1=0.0)
    for el in stash:
        i=el[0]
        j=el[1]
        ang1 = j*phi
        ang2 = j*phi
        if(j*phi+phi/2 > np.pi/2): ang1 = np.pi/2 - phi/2
        elif(j*phi-phi/2 < -np.pi/2): ang2 = np.pi/2 + phi/2
        a1=2*np.pi*Rorb*np.cos(ang1+phi/2)/n
        a2=2*np.pi*Rorb*np.cos(ang2-phi/2)/n
        a=2*np.pi*Rorb*np.cos(j*phi)/n
        coords=np.array([[a1/2,a/2,0],[-a1/2,a/2,0],[-a2/2,-a/2,0],[a2/2,-a/2,0]])
        meg = dy.Megastructure(Rorb*np.cos(j*phi), False, isrot=True,elevation=Rorb*np.sin(j*phi), Plcoords=coords, ph_offset=i*phi)
        
        meg.Plcoords=meg.rotate([1,0,0],-j*phi)
        meg.Plcoords=meg.rotate([0,1,0],i*phi)
        sim1.add_megs(meg)
    #possibilities = np.delete(possibilities, np.arange(0,track_ind), axis = 0)
    #print("check total:", len(possibilities))
        
    print("Accepted:",len(sim1.megs), len(stash))
    road, ph, lc = sim1.simulate_transit()
    print("--- %s seconds ---" % (time.time() - start_time))
    return(lc)




#--------------------------------------------------------------------
pos = list_of_places()
sh = one_arbitrary_shape(np.array([[0,0],[1,0],[0,-1]]),8,pos)
x = translate_or_flip(np.array([[0,0],[1,0],[0,-1]]), np.array([[0,0],[1,0],[0,1]]),pos)
print(x)
plot_shape([[0,0],[1,0],[0,-1]])
plot_shape([[0,0],[1,0],[0,1]])

#--------------------------------------------------------------------

#redo the animation frame coz we need a progressive animation showing
#evolution of dyson swarm

#plt.style.use('dark_background')


# plt.rcParams["font.family"] = "serif"
# fig = plt.figure(figsize=(7,7))
# fig.patch.set_facecolor('#CCCCCC')
# ax1 = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
# ax2 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
# ax3 = plt.subplot2grid((3, 3), (0, 2), rowspan=2)
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro')
# th = np.linspace(0,2*np.pi, 200)

# temp = np.array([[x['x'] for x in el] for el in road.traj])
# maxorb = max(np.abs(temp.reshape(-1)))
# thframe = np.linspace(-np.pi,np.pi,no_pt)

# fr_sum = np.tile(np.arange(0,no_pt,3), maxout)



# def init():
#     ln.set_data([], [])
#     ax2.set_xlabel('Phase')
#     ax2.set_ylabel('Flux')
#     ax2.set_xlim(-np.pi, np.pi)
#     ax2.set_ylim(-0.1,1.3)
#     ax3.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
#     plt.suptitle('Dyson Swarm in Construction')
#     plt.figtext(0.45, 0.93, "@ushasibhowmick", fontsize=8, color='#F33434')

#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#     ax1.spines['bottom'].set_visible(False)
#     ax1.spines['left'].set_visible(False)

#     ax3.spines['top'].set_visible(False)
#     ax3.spines['right'].set_visible(False)
#     ax3.spines['bottom'].set_visible(False)
#     ax3.spines['left'].set_visible(False)

#     props = dict(boxstyle='round', facecolor='black', alpha=0.5, pad=1)
#     txt = "Main Panel: "+str(np.round(2*np.pi*Rorb/(n*Rstar),2))+"$R_{st}$\nOrbit: "+str(Rorb/Rstar)+"$R_{st}$\nu: "+str(0
#         )+"\ne: "+str(0)

#     ax3.text(0.5, 0.3, txt, fontsize=9,transform=ax3.transAxes,  horizontalalignment='center',
#             verticalalignment='center', linespacing=2, bbox=props, color='white')

#     ax2.set_ylabel('Flux')
#     ax2.set_xlabel('Phase')
#     theta = np.arange(-np.pi, np.pi+np.pi/4, step=(np.pi / 4))
#     ax2.set_xticks(theta)
#     ax2.set_xticklabels([ '-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])
#     return ln,

# it = -1
# def update(frame):
#     global it
    
#     if(frame==0): 
#         it+=1
#         ax2.clear()
#         ax2.set_ylim(-0.1,1.1)
#         #ax2.grid(which="major",alpha=0.6, color='gray', ls=':')
#         ax2.set_ylabel('Flux')
#         ax2.set_xlabel('Phase')
#         theta = np.arange(-np.pi, np.pi+np.pi/4, step=(np.pi / 4))
#         ax2.set_xticks(theta)
#         ax2.set_xticklabels([ '-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])
#     if(it==maxout): it = 0

#     props = dict(boxstyle='round', facecolor='black',edgecolor='#ffa343', pad=1)
#     ax3.text(0.5, 0.8, "No. of Panels:\n"+str(len(sum_road[it].traj)), fontsize=9,transform=ax3.transAxes,  horizontalalignment='center',
#             verticalalignment='center', linespacing=2, bbox=props, color='#ffa343')

#     ax1.clear()
#     zst=1
#     zpl=[0 if np.all(el[frame]['z']<0) else 3 for el in sum_road[it].traj]
#     ax1.set_aspect(1)
#     ax1.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
#     ax1.fill(sum_road[it].Rstar*np.cos(th), sum_road[it].Rstar*np.sin(th), zorder = zst, color='#ffa343')
#     ax1.scatter(road.MCscatter_x, road.MCscatter_y,  marker='.', s=1,color='#fff44f', zorder=2)
#     ax1.set_xlim(-maxorb*1.2,maxorb*1.2)
#     ax1.set_ylim(-maxorb*1.2,maxorb*1.2)

#     i=0
#     for el in sum_road[it].traj:
#         if(it == 0):
#             ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='#0E092C', edgecolor='gray', alpha=0.8)
#         elif(i>=len(sum_road[it-1].traj)):
#             ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='#0E092C', edgecolor='gray', alpha=0.8)
#         else:
#             ax1.fill(el[frame]['x'],el[frame]['y'], zorder=zpl[i], color='#181818', edgecolor='gray', alpha=0.9)
#         i+=1
#     ax2.plot(thframe, sum_lc[it], color='#F33434')
#     return ln,


# #net = np.array([np.array([thframe[i]]+np.array(sum_lc)[:,i]) for i in range(no_pt)])
# net =np.transpose(np.insert(np.array(sum_lc),0, thframe, axis=0))
# ani = animation.FuncAnimation(fig, update, frames=fr_sum, interval=1,
#                     init_func=init)

# print(np.array(net).shape)
# np.savetxt('sphere_conc_org_1.csv',net,delimiter=' ', header='phase, panels:1,2,4,8,16...')
# writergif = animation.PillowWriter(fps=15) 
# ani.save('completed_sphere_org_1.gif', writer=writergif, savefig_kwargs=dict(facecolor='#CCCCCC'))

# #plt.show()