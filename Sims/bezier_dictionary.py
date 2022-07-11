import random
import numpy as np
import matplotlib.pyplot as plt
import aliensims as dysim
import os
import time
import pandas as pd
from transit import occultnonlin, occultquad
from multiprocessing import Process, Pool
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from bezier import get_random_points, get_bezier_curve
import h5py
import warnings
warnings.filterwarnings("ignore")

start_time = time.time()
bez_shape=[]
ca2=['#432371', '#714674', '#9F6976', '#CC8B79', '#FAAE7B']
Rorb = 0
res = 0
fl = np.pi/2

def shape_dictionary():
    np.random.seed(101010101)
    rads = np.linspace(0,1,6)
    edges = np.sqrt(1/np.linspace(0.1,0.9,5) -1)
    ns = [5,6,7,8,9,10,11,12,13,14]

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
                for num in range(20):
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

def bezier_sim(x):
    global bez_shape
    global Rorb
    global res
    np.random.seed(3456*x)
    sim_2d = dysim.Simulator (1, res, 500, limb_u1=0.0, limb_u2=0.0)
    meg_2d = dysim.Megastructure(Rorb, iscircle=True, Rcircle=0.3, isrot=True)
    # meg_2d.Plcoords = np.array(bez_shape)
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
        with Pool(processes=4) as pool:
            output = np.asarray(pool.map(bezier_sim, range(16)))
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

        # sim_2d = dysim.Simulator (1, 1000, 700, limb_u1=0.0, limb_u2=0.0)
        # meg_2d = dysim.Megastructure(2, isrot=True)
        # meg_2d.Plcoords = np.array(bez_shape)
        # sim_2d.add_megs(meg_2d)
        # # sim_2d.set_frame_length()
        # sim_2d.simulate_transit()
        # TA = dysim.Transit_Animate(sim_2d.road, sim_2d.megs, lc2d, sim_2d.frames)
        # TA.go(ifsave=True,filepath='testbezier.gif')

        # plt.style.use('seaborn-bright')
        # fig, ax = plt.subplots(1,1, figsize = (20,7), sharex=True)

        # frm = np.linspace(-fl,fl, len(lc2d))
        # ax.plot(frm,lc2d,label = '2d')
        # ax.legend()
        # ax.set_ylabel('Flux')
        # ax.set_title("Bezier Transit")
        # plt.suptitle('2D vs 3D transiting objects')

        # df = pd.DataFrame(zip(frm, lc2d, lc2dstd), columns=['frame','flux', 'std'])
        # df.to_csv('solarsim.csv', index='False', sep=',')
        #np.savetxt('2d3d_0.1R_circ.csv', np.transpose(np.array([frm, lc2d, lc2dstd, lc3d, lc3dstd])),delimiter=',', header='frame, 2d, 2dstd, 3d, 3dstd')
        # plt.savefig('solarsim.png')

    return(lc_list, lc_std_list, frmlist)

def run_one_bezier_sim(shapes, orbit, resolution):
    global bez_shape
    global Rorb
    global res
    
    res = resolution
    bez_shape = shapes
    Rorb = orbit
    with Pool(processes=40) as pool:
        output = np.asarray(pool.map(bezier_sim, range(80)))
        lc2dsum=[np.array(el[0]) for el in output]
        lc2d = np.mean(lc2dsum, axis = 0)
        fl = np.mean(np.array([el[1] for el in output]))
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        print("--- %s min ---" % ((time.time() - start_time)/60))
    frm = np.linspace(-fl,fl, len(lc2d))
    return(lc2d, lc2dstd, frm)


#code run
shape_entries = os.listdir('../Shape_Directory/shape_list/')


# for entry in [shape_entries[7],shape_entries[8],shape_entries[9],shape_entries[1]]:
#     print(entry)
#     hf = h5py.File("../Shape_Directory/shape_list/"+entry, 'r')
#     f1 = h5py.File("../Shape_Directory/shape_lc/"+entry, "w")
#     fig, ax = plt.subplots(1,1,figsize=(10,10))
#     i=int(0)
#     for k1 in hf:
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
    
#     hf.close()
#     f1.close()
#     plt.savefig("../Shape_Directory/shape_lc/"+entry[:-5]+'_'+str(i)+'.png')
    #plt.show()


hf = h5py.File("../Shape_Directory/shape_lc/n_5.hdf5", 'r')
print(len(hf), shape_entries)