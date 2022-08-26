import random
import numpy as np
import matplotlib.pyplot as plt
# import aliensims as dysim
import os
import time
import pandas as pd
from transit import occultnonlin, occultquad
from multiprocessing import Process, Pool, cpu_count
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from bezier import get_random_points, get_bezier_curve
import h5py
import warnings
warnings.filterwarnings("ignore")

"""LIBRARY OF ARBITRARY SHAPES: THE BEZIER WAY
In this module we use bezier curves to generate a bunch of arbitrary shapes. These shapes 
will be utilized to obtain corresponding LC. Hopefully we will be able to train a NN to reverse-
engineer this. 

"""

PROCESS_RANGE = 70
CORES = cpu_count()

start_time = time.time()
bez_shape=[]
ca2=['#432371', '#714674', '#9F6976', '#CC8B79', '#FAAE7B']
Rorb = 0
res = 0
fl = np.pi/2

def shape_dictionary(ns, no_of_rads=6, no_of_edges=5, no_in_each_config = 20):
    """ This is a function meant to generate a number of random bezier shapes and use them for
        further analysis. Total number of shapes this generates is 
        ns*no_of_rads*no_of_edges*no_in_each_config

        :param ns: array of n values - an n value determines the number of vertices in the shape
        :param no_of_rads: divide the rad parameter space into an array of these many values
        :param no_of_edges: divide the edgy parameter space into an array of these many values
        :param no_in_each_config: number of random shapes generated per rad edgy config


    """

    if not os.path.exists('../Shape_Directory/shape_list/'):
        os.mkdir('../Shape_Directory/shape_list/')

    if not os.path.exists('../Shape_Directory/shape_plot/'):
        os.mkdir('../Shape_Directory/shape_plot/')

    np.random.seed(101010101)
    rads = np.linspace(0,1,no_of_rads)
    edges = np.sqrt(1/np.linspace(0.1,0.9,no_of_edges) -1)
    # ns = [5,6,7,8,9,10,11,12,13,14]

    grid_size= 10
    grid=np.array([[[i,j]  for i in range(grid_size)] for j in range(grid_size)]).reshape(grid_size**2,2)
    
    for n in ns:
        fig, axs = plt.subplots(3,2, figsize=(8,10))
        plt.suptitle('n = '+str(n))
        f1 = h5py.File("../Shape_Directory/shape_list/n_"+str(n)+".hdf5", "w")
        for rad,ax in zip(rads,axs.ravel()):
            count = 0
            ax.set_aspect("equal")
            ax.tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
            ax.set_title("rad:"+str(np.around(rad,2)))
            col_cnt = 0
            for edgy in edges:
                for num in range(no_in_each_config):
                    a = get_random_points(n=n, scale=0.8) 
                    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
                    x = x - np.mean(x)
                    y = y - np.mean(y)
                    shape = [[i,j,0] for i,j in zip(x,y)]
                    dset1 = f1.create_dataset("rad_"+str(np.around(rad,2))+'_edg_'+str(np.around(edgy,2))+'_'+str(num), np.array(shape).shape, dtype='f', data=shape)

                    a = a + grid[count]/1.2
                    x,y, _ = get_bezier_curve(a,rad=rad, edgy=edgy)
                    if(num==0): 
                        ax.plot(x,y, color=ca2[col_cnt], label=str(np.around(edgy,2)))
                    else: ax.plot(x,y, color=ca2[col_cnt])
                    count+=1
                col_cnt+=1
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig('../Shape_Directory/shape_plot/n_'+str(n)+'.png')
        plt.close()
        f1.close()

        print('n: ', n, " ...created!")

def bezier_sim(x):
    global bez_shape
    global Rorb
    global res
    np.random.seed(3456*x)
    sim_2d = dysim.Simulator (1, res, 300, limb_u1=0.0, limb_u2=0.0)
    meg_2d = dysim.Megastructure(Rorb, isrot=True)
    meg_2d.Plcoords = np.array(bez_shape)
    sim_2d.add_megs(meg_2d)
    sim_2d.set_frame_length()
    sim_2d.simulate_transit()
    return(sim_2d.lc, sim_2d.frame_length)

def run_bezier_sim(shapes_list, orbit_list, resolution):
    global bez_shape
    global Rorb
    global res
    lc_list=[]
    lc_std_list=[]
    frmlist = []
    res = resolution
    for shape,orb in zip(shapes_list, orbit_list):
        bez_shape = shape
        Rorb = orb
        with Pool(processes=CORES) as pool:
            output = np.asarray(pool.map(bezier_sim, range(PROCESS_RANGE)))
            lc2dsum=[np.array(el[0]) for el in output]
            print(np.array(lc2dsum).shape)
            lc2d = np.mean(lc2dsum, axis = 0)
            fl = np.mean(np.array([el[1] for el in output]))
            lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
            print("--- %s min ---" % ((time.time() - start_time)/60))
        lc_list.append(lc2d)
        lc_std_list.append(lc2dstd)
        frm = np.linspace(-fl,fl, len(lc2d))
        frmlist.append(frm)

    return(lc_list, lc_std_list, frmlist)

def run_one_bezier_sim(shapes, orbit, resolution):
    """ Run a bezier simulation for a 
    """
    global bez_shape
    global Rorb
    global res
    
    res = resolution
    bez_shape = shapes
    Rorb = orbit
    with Pool(processes=CORES) as pool:
        output = np.asarray(pool.map(bezier_sim, range(PROCESS_RANGE)))
        lc2dsum=[np.array(el[0]) for el in output]
        lc2d = np.mean(lc2dsum, axis = 0)
        fl = np.mean(np.array([el[1] for el in output]))
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        # print("--- %s min ---" % ((time.time() - start_time)/60))
    frm = np.linspace(-fl,fl, len(lc2d))
    return(lc2d, lc2dstd, frm)

def analyse_bezier_results():
    """ Ran the above simulations and hope to see if any of the bezier shapes made any difference. 

    """

    hf = h5py.File("../Shape_Directory/shape_lc/n_14.hdf5", 'r')
    i = 0
    fig, ax = plt.subplots(2,1, figsize=(10,10))
    old=[]
    for key in hf:
        if(i==50): break
        if(key.find('lc')>0):
            i+=1
            n = np.array(hf.get(key))
            ax[0].plot(n, label=key)
            if(len(old)>0): 
                ax[1].plot(old-n)
                old = n
            else: old = n
    ax[0].legend()
    plt.show()
  

def plot_sims(no, file):
    """ The earlier plots were a mistake, forgot to uncomment something... now 
    we're gonna just plot some results...
    
    :param no: how many samples
    :param file: file number (1 for file n_1.hdf5, 2 for file n_2.hdf5 etc.)
    """
    ca2=['#432371', '#714674', '#9F6976', '#CC8B79', '#FAAE7B']

    hf = h5py.File("../Shape_Directory/shape_lc/n_"+str(file)+".hdf5", 'r')
    i = 0
    fig, ax = plt.subplots(no,2, figsize=(10,12), gridspec_kw={ 'width_ratios': [3,1],
        'wspace': 0.01,'hspace': 0.1})
    
    for key in hf:
        if(key.find('sh')>0): 
            ax[i][1].set_aspect('equal', adjustable='box')
            sh = np.array(hf.get(key))
            ax[i][1].fill(sh[:,0], sh[:,1],color=ca2[i%5])
            frm = np.array(hf.get(key[:-2]+'frm'))
            lc =  np.array(hf.get(key[:-2]+'lc'))
            ax[i][0].plot(frm, lc, color=ca2[4 - i%5])
            ax[i][1].tick_params(left = False, right = False , labelleft = False ,labelbottom = False, bottom = False)
            if(i<no-1): ax[i][0].tick_params(labelbottom = False, bottom = False)
            ax[i][1].set_xlim(-0.5,0.5)
            ax[i][1].set_ylim(-0.5,0.5)
            ax[i][0].set_ylabel('Flux', size=13)

            i+=1
        if(i==no): break
    # ax[0].legend()
    ax[no-1][0].set_xlabel('Phase', size=13)
    plt.suptitle('Simulation: n = '+str(file), size=15)
    plt.tight_layout()
    plt.savefig("sample_from_n"+str(file)+".png")
    plt.show()

def select_1000_shapes(no_of_el_filt, filename='filtered_list.hdf5'):
    """ function to select the some 10 shapes from each directory in shape list. Now we use these 
    to iterate over and create a grid.
    
    :param no_of_el_filt: number of shapes to filter to create the grid (in this case 1000)
    :param filename: name of the output hdf5 file where we store this.

    """

    shape_entries = os.listdir('../Shape_Directory/shape_list/')
    chlen = len(shape_entries)

    hf = h5py.File("../Shape_Directory/shape_list/"+shape_entries[0], 'r')
   
    np.random.seed(100100)
    choice = np.arange(0,len(hf),1)
    np.random.shuffle(choice)
    choice = choice[:int(no_of_el_filt/chlen)]

    tabs = 0
    f1 = h5py.File("../Shape_Directory/"+filename, "a")
    for entry in shape_entries:
        print(entry,'...filtering')
        hf = h5py.File("../Shape_Directory/shape_list/"+entry, 'r')
        i=0
        
        for ki in hf:
            if((choice==i).any()):
                n = np.array(hf.get(ki))
                dset1 = f1.create_dataset('sh'+str(tabs), np.array(n).shape, dtype='f', data=n)
                dset1.attrs['n_val'] = entry[:-5]
                dset1.attrs['head'] = str(ki)
                tabs+=1
            i+=1


def one_config_1000_shapes(rorb, scale, filename='filtered_list.hdf5'):
    """ This is an attempt to build the extended dictionary... lets see how many we get through. This 
    function simulates the lightcurve for all the shapes in an hdf5 list created by the select_1000_shapes
    function for one configuration of rorb and scale.

    :param rorb: radius of orbit
    :param scale: scaling of the shape

    """
    if not os.path.exists('../Shape_Directory/shape_grid/'):
        os.mkdir('../Shape_Directory/shape_grid/')

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    f1 = h5py.File("../Shape_Directory/"+filename, "r")
    i=0

    if(scale>0.799): res = 5000
    elif(scale>0.5999): res = 6000
    elif(scale>0.399): res = 8000
    else: res = 12000
    print('res:',res)
    for ki in f1:
        sh = np.array(f1.get(ki))
        
        i+=1
        
        
        
        if(i%50==0):
            plt.savefig("../Shape_Directory/shape_grid/orb"+str(np.around(rorb,2))+'_scale'+str(np.around(scale,2))+'_'+str(i)+'.png')
            plt.close()
            fig, ax = plt.subplots(1,1,figsize=(10,10))
            print("--- %s min ---" % ((time.time() - start_time)/60))

        lc, lcstd, frm = run_one_bezier_sim(sh*scale, rorb, res)
            
        try: df = pd.read_csv('../Shape_Directory/shape_grid/Rorb_'+str(np.around(rorb,2))+'scl_'+str(np.around(scale,2))+'.csv', sep=',')
        except:
            # print("check in",str(ki)+'_frm')
            df = pd.DataFrame(zip(frm, lc), columns=[str(ki)+'_frm',str(ki)+'_lc'])
            df.to_csv('../Shape_Directory/shape_grid/Rorb_'+str(np.around(rorb,2))+'scl_'+str(np.around(scale,2))+'.csv', index=False,sep=',')
            continue
        df[str(ki)+'_frm']=frm
        df[str(ki)+'_lc']=lc
        print("check out",str(ki), rorb, scale)
        df.to_csv('../Shape_Directory/shape_grid/Rorb_'+str(np.around(rorb,2))+'scl_'+str(np.around(scale,2))+'.csv', index=False,sep=',')
        ax.plot(frm, lc)

#----------------------------------------------------------------------------

shape_dictionary([5,6,7,8,9,10,11,12,13,14])
select_1000_shapes(20, 'filtlist.hdf5')

scale_arr=np.array([0.2,0.4,0.6,0.8,1.0])
rorb_arr=np.around(np.logspace(0.31,2,5), 2)

for sc in scale_arr[4:]:
    for rorb in rorb_arr[:1]:
        one_config_1000_shapes(rorb, sc)

# select_1000_shapes()


#1.0: 5000    0.8: 5000     0.6:  6000     0.4: 8000      0.2:  12000    

#----------------------------------------------------------------------------

# shape_entries = os.listdir('../Shape_Directory/shape_list/')

# for entry in shape_entries[8:]:
#     print(entry)
#     hf = h5py.File("../Shape_Directory/shape_list/"+entry, 'r')
#     # f1 = h5py.File("../Shape_Directory/shape_lc/"+entry, "a")
#     fig, ax = plt.subplots(1,1,figsize=(10,10))
#     i=int(0)
#     for k1 in hf:
#         # if(i<400): 
#         #     i+=1
#         #     continue
#         f1 = h5py.File("../Shape_Directory/shape_lc/"+entry, "a")
#         i+=1
#         if(i%10==0):
#             plt.savefig("../Shape_Directory/shape_lc/"+entry[:-5]+'_'+str(i)+'.png')
#             plt.close()
#             fig, ax = plt.subplots(1,1,figsize=(10,10))
#         n = np.array(hf.get(k1))
#         print(k1, n.shape)
#         lc, lcstd, frm = run_one_bezier_sim(n, 2, 5000)
#         dset1 = f1.create_dataset(str(k1)+'_sh', np.array(n).shape, dtype='f', data=n)
#         dset2 = f1.create_dataset(str(k1)+'_frm', np.array(frm).shape, dtype='f', data=frm)
#         dset3 = f1.create_dataset(str(k1)+'_lc', np.array(lc).shape, dtype='f', data=lc)
#         ax.plot(frm, lc)
#         f1.close()


#----------------------------------------------------------------------------------------

# #Diagnostic section
# shape_entries = os.listdir('../Shape_Directory/shape_list/')
# hf = h5py.File("../Shape_Directory/shape_lc/n_5.hdf5", 'r')
# print(len(hf), shape_entries)

#----------------------------------------------------------------------------------------
