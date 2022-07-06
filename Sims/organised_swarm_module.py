import random
import h5py
#from turtle import color, pos
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dy
import time
from multiprocessing import Process, Pool

start_time = time.time()

"""DYSON SPHERE IN CONSTRUCTION
This is a module that can generate all manner of possibilities that may arise during the construction 
of a dyson sphere, panel by panel. When run to its full potential, this will lead to a set of more than
a million possibilities... however this is stashed and replaced by a more convenient arbitrary shape 
generator. Its still a very well written code.

Here we define an 'n' value, which is the number of panels you want to put in one orbit around a star.
This means that an n=10 means that one equator of the star takes 10 panels to cover it.

Other parameters:
    Rstar: Radius of the star... doesnt matter what you give it, as long as Rorb is given accordingly
    Rorb: Radius of the orbit. Distance at which the swarm is built
    no_pt: Number of frames that one revolution around the star would contain
    res: Resolution of the Monte-Carlo Simulation
    maxout: Number of iterations. If this is less than the no. of iterations needed to complete the 
        sphere, then the animation stops early. Else the animation stops at the completiono of the 
        sphere.

"""

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
    """An initializer function that generates an array containing the location of each panel of the completed
    dyson sphere. A shape will be a subset of this list.
    
    """
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
    #print(phi)
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
        if(np.any(meg.Plcoords[:,2]<0)):
            ax.fill(meg.Plcoords[:,0],meg.Plcoords[:,1],color='#000000', edgecolor='gray', alpha=0.8, zorder=0)
        else:
            ax.fill(meg.Plcoords[:,0],meg.Plcoords[:,1],color='#000000', edgecolor='gray', alpha=0.8, zorder=2)
    # plt.show()
        
#just adding a function to plot a large number of shapes together, so that we can look at them more easily
#later
def plot_shapes(shapes, numx, numy):
    fig, axs = plt.subplots(numx, numy, figsize=(numy*2,numx*2))
    th = np.linspace(0,2*np.pi,200)
    #print(phi)
    for shape, ax in zip(shapes, axs.ravel()):
        ax.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
        ax.set_aspect('equal', adjustable='box')
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
            if(np.any(meg.Plcoords[:,2]<0)):
                ax.fill(meg.Plcoords[:,0],meg.Plcoords[:,1],color='#000000', edgecolor='gray', alpha=0.8, zorder=0)
            else:
                ax.fill(meg.Plcoords[:,0],meg.Plcoords[:,1],color='#000000', edgecolor='gray', alpha=0.8, zorder=2)

#now we need to do something about the degeneracies.
def translate_or_flip(shape, nextsh, possibilities):
    """ This is a degeneracy checking function that checks if two shapes are degenerate. Degenerate shapes
    are those which are identical when translated along the x-axis, or flipped along the y-axis.

    :param shape: One of the two shapes for which we need to check for degeneracies
    :param nextsh: One of the two shapes for which we need to check for degeneracies
    :param possibilities: wholesome array of possibilities returned from the list of places function

    Returns -
    1 if degenerate, 0 if non-degenerate.
    
    """
    xmax = max(possibilities[:,0])
    xmin = min(possibilities[:,0])
    ymin = min(possibilities[:,1])
    ymax = max(possibilities[:,1])

    shapemaxx=max(shape[:,0])
    shapeminx=min(shape[:,0])

    #identical from different seeds
    # print(shape, nextsh)
    if(len(shape)!=len(nextsh)): return(0)
    if(np.any([(shape==el).all(axis=1) for el in nextsh], axis=1).all()): return(1)
    
    #setting range... still some fixes here
    rng = shapemaxx-shapeminx
    if((rng)>len(shape)): rng = xmax+1 - rng

    #translate along x
    for i in range(-rng,rng+1,1):
        newshape=shape+[i,0]
        #print(newshape, shape)
        if(np.any(newshape[:,0]>xmax)):
            newshape[np.where(newshape>xmax)[0][0],0] = newshape[np.where(newshape>xmax)[0][0],0] - xmax -1
        if(np.any(newshape[:,0]<xmin)):
            newshape[np.where(newshape<xmin)[0][0],0] = newshape[np.where(newshape<xmin)[0][0],0] + xmax +1

        # print(newshape, nextsh)
        # print(np.any([(newshape==el).all(axis=1) for el in nextsh], axis=1).all())
        if(np.any([(newshape==el).all(axis=1) for el in nextsh], axis=1).all()): return(1)

    #flip along y
    shape2=np.array([[el[0], -el[1]] for el in shape])
    
    
    if(np.any([(shape2==el).all(axis=1) for el in nextsh], axis=1).all()): return(1)

    for i in range(-rng,rng+1,1):
        newshape=shape2+[i,0]
        #print(newshape, shape)
        if(np.any(newshape[:,0]>xmax)):
            newshape[np.where(newshape>xmax)[0][0],0] = newshape[np.where(newshape>xmax)[0][0],0] - xmax -1
        if(np.any(newshape[:,0]<xmin)):
            newshape[np.where(newshape<xmin)[0][0],0] = newshape[np.where(newshape<xmin)[0][0],0] + xmax +1

        # print(newshape, nextsh)
        # print(np.any([(newshape==el).all(axis=1) for el in nextsh], axis=1).all())
        if(np.any([(newshape==el).all(axis=1) for el in nextsh], axis=1).all()): return(1)

    else: return 0


#we need a function to add a layer of panels over the original layers
def layers_over_shape(shape,possibilities):
    xmax = max(possibilities[:,0])
    xmin = min(possibilities[:,0])
    ymin = min(possibilities[:,1])
    ymax = max(possibilities[:,1])
    ind = np.array([np.where((possibilities == np.array(el)).all(axis=1))[0] for el in shape]).reshape(-1)
    possibilities_left=np.delete(possibilities,ind, axis=0)
    count=0
    layers=[]
    for el in possibilities_left:
        if(np.any((np.array(shape)==np.array([1,0])+el).all(axis=1))):  layers.append(el)
        elif(np.any((np.array(shape)==np.array([-1,0])+el).all(axis=1))): layers.append(el)
        elif(np.any((np.array(shape)==np.array([0,1])+el).all(axis=1))): layers.append(el)
        elif(np.any((np.array(shape)==np.array([0,-1])+el).all(axis=1))): layers.append(el)
        elif((np.array([1,0])+el)[0]>xmax or (np.array([-1,0])+el)[0]<0):
            if(np.any((np.array(shape)==np.array([-xmax,0])+el).all(axis=1))): layers.append(el) 
            if(np.any((np.array(shape)==np.array([xmax,0])+el).all(axis=1))): layers.append(el)
    return(np.array(layers))



#here's a rerun of the simulation... testing for a simple case... we'll start off with one panel... then
#move on to the next iteration. each iteration is one panel added to the preexisting shape. This is the 
#least complex way to do this.
def run_shape_generator(iterations):
    """ This is the main running function, that generates(hopefully) all the possibilities that constitute
    building a dyson sphere to completion. It takes in a number of iterations, and generate all distinct 
    shapes from it. Degeneracies are removed. It returns the directory of shapes in the form of an array 
    of shapes. Each shape is a specific list of possibilities from the 'list of places' array. The directory 
    is also stored as 'data.hdf5', and a plot is made demonstrating the shapes.

    :param iterations: This is the number of iterations the code will run for. Each iteration adds a panel
    so 3 iterations means 4 panels.

    Returns - 
    Directory of shapes

    """
    f1 = h5py.File("data.hdf5", "w")
    pos = list_of_places()
    #how many panels to add next?
    i=0
    directory_of_shapes_init= [[[0,0]]]
    while(i<iterations):
        i+=1
        directory_of_shapes = list(np.copy(directory_of_shapes_init))
        for strt in directory_of_shapes_init:
            
            sum_of_layers = layers_over_shape(strt, pos)
            for el in sum_of_layers:
                newshape = np.array(np.append(strt,np.array([el]), axis=0))
                degeneracy = [translate_or_flip(one, newshape, pos) for one in directory_of_shapes]
                if(np.any(np.array(degeneracy))): 
                    continue
                else: 
                    # print(newshape,np.array(directory_of_shapes, dtype='object'), degeneracy)
                    directory_of_shapes.append(newshape)
        directory_of_shapes_init = directory_of_shapes
    
    print(len(directory_of_shapes_init))

    i=0
    for el in directory_of_shapes_init:
        dset1 = f1.create_dataset("sh_"+str(i), np.array(el).shape, dtype='i', data=el)
        i+=1
        #print(el)
    f1.close()

    plot_shapes(directory_of_shapes_init,6,7)
    plt.savefig('fig.png')
    return(directory_of_shapes)
   
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
# pos = list_of_places()
# sh = one_arbitrary_shape(np.array([[0,0],[1,0],[0,-1]]),8,pos)
# print('layers',layers_over_shape(np.array([[59,0],[59,-1]]),pos))
# x = translate_or_flip(np.array([[0,0],[0,1],[1,0]]), np.array([[0,0],[1,0],[0,-1]]),pos)
# print(x)
# plot_shape([[59,0],[59,-1]])
# # plot_shape([[0,0],[1,0],[0,1]])

run_shape_generator(3)

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