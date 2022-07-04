import random
from traceback import format_list

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

"""TESTING MODULES
    This is primarily used for testing out the aliensims.py package and creating sims for varoius scenarios
    Each function corresponds to one scenario and is used for various miscellaneous tasks. Primarily meant
    for running on the server, therefore optimized for parallel processing routines.
"""

#-------------------------------------------------------------------------
#yo! kumaran recieving? over and out.
# flip , 120
#50: 1500   45:2000  39:2500  34:3000  28:3500  23:4000  17:5000  12:6000
#6: 12000   #1:20000
#-------------------------------------------------------------------------

# These global variables are mainly for the 2d vs 3d simulations
testg = 0
start_time = time.time()
fl = np.pi/3
sides = 3

Rpl=39
Rorb=200
u1=0
u2=0
Rstar=100
b=0
ecc=0
per_off=0 

res = 2000


#==================================================================================

def new_plar(ph,p,u1,u2,rorb,imp):
    """ The function that returns the Mandel and Agol formulation of a transit feature
    """
    znp = np.sqrt(np.abs(rorb*np.sin(ph*np.pi))**2+imp**2)
    a= occultquad(znp,p,[u1,u2])  
    return(a -1) 

def test_multi_loops_3d(x):
    """ 3D function to be put in parallel processor later in function twoD_vs_threeD
    """
    global testg
    global Rpl
    global u1
    global u2
    global b
    global fl
    global res
    global Rorb
    np.random.seed(1234*x)
    sim_3d = dysim.Simulator (Rstar, res, 500, fl, limb_u1=u1, limb_u2=u2)
    meg_3d = dysim.Megastructure(Rorb, True, Rpl, incl=np.arcsin(b*Rstar/Rorb), per_off=0, ecc=0.0)
    sim_3d.add_megs(meg_3d)
    sim_3d.set_frame_length()
    sim_3d.simulate_transit()
    fl=sim_3d.frame_length
    #if(x==0): print("Count:", meg_3d.set, x, np.pi/sim_3d.frame_length)
    return(sim_3d.lc)

def test_multi_loops_2d(x):
    """ 2D function to be put in parallel processor later in function twoD_vs_threeD
    """
    global testg
    np.random.seed(3456*x)
    global Rpl
    global u1
    global u2
    global b
    global fl
    global res
    global Rorb
    sim_2d = dysim.Simulator (Rstar, res, 500, fl, limb_u1=u1, limb_u2=u2)
    meg_2d = dysim.Megastructure(Rorb, True, Rpl, incl=np.arcsin(b*Rstar/Rorb), isrot=True, per_off=0, ecc=0)
    sim_2d.add_megs(meg_2d)
    sim_2d.set_frame_length()
    sim_2d.simulate_transit()
    fl=sim_2d.frame_length
    #if(x==0): print("Count:", meg_3d.set, x, np.pi/sim_3d.frame_length)
    return(sim_2d.lc)

def fit_alg(phl, rpll, rorbl, bl):
    """ Stash a fitting algorithm using the simulator instead of the grid... for fine tuning and 
        verification of grid parameters.
    """
    global Rpl
    global u1
    global u2
    global b
    global fl
    global Rorb
    Rpl = rpll*Rstar
    u1 = 0.3975
    u2 = 0.2650
    b = bl
    Rorb = rorbl*Rstar
    with Pool(processes=40) as pool:
        lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(80)))
        lc2d = np.mean(lc2dsum, axis = 0)
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        #print("--- %s min ---" % ((time.time() - start_time)/60))
    
    phin = np.linspace(-fl,fl,500)
    print('go')
    val = interp1d(phin, lc2d, kind='linear', fill_value='extrapolate')(phl*np.pi)
    return(val-1)

def fit_alg_3d(phl, rpll, rorbl, bl):
    """ Stash a fitting algorithm using the simulator instead of the grid... for fine tuning and 
        verification of grid parameters.
    """
    global Rpl
    global u1
    global u2
    global b
    global fl
    global Rorb
    Rpl = rpll*Rstar
    u1 = 0.3975
    u2 = 0.2650
    b = bl
    Rorb = rorbl*Rstar
    with Pool(processes=40) as pool:

        lc3dsum = np.asarray(pool.map(test_multi_loops_3d, range(80)))
        # print("--- %s min ---" % ((time.time() - start_time)/60))

        lc3d = np.mean(lc3dsum, axis = 0)
        lc3dstd = np.sqrt(np.mean((lc3dsum-lc3d)**2, axis=0))
    
    phin = np.linspace(-fl,fl,300)
    print('go')
    val2 = interp1d(phin, lc3d, kind='linear', fill_value='extrapolate')(phl*np.pi)
    return(val2-1)

def twoD_vs_threeD():
    """ Main thing that runs on the server... generate a 2D and 3D LC of same dimensions and distance from the 
        star using multiprocessing - server compatible code to generate high resolution lightcurves."""
    global fl
    global testg
    # start 4 worker processes
    with Pool(processes=40) as pool:
        lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(120)))
        lc2d = np.mean(lc2dsum, axis = 0)+np.flip(np.mean(lc2dsum, axis = 0))
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        print("--- %s min ---" % ((time.time() - start_time)/60))

        lc3dsum = np.asarray(pool.map(test_multi_loops_3d, range(120)))
        print("--- %s min ---" % ((time.time() - start_time)/60))

        lc3d = np.mean(lc3dsum, axis = 0)+np.flip(np.mean(lc3dsum, axis = 0))
        lc3dstd = np.sqrt(np.mean((lc3dsum-lc3d)**2, axis=0))

        mn = (np.asarray(lc3d-lc2d)**2).sum()/len(lc3d)
        print(np.sqrt(mn))

        plt.style.use('seaborn-bright')
        fig, ax = plt.subplots(2,1, figsize = (7,10), sharex=True)

    frm = np.linspace(-fl,fl, len(lc3d))
    print("n",fl)
    #model = new_plar(frm/np.pi,0.01, 0,0,2,0)+1
    ax[0].plot(frm,lc2d,label = '2d')
    ax[0].plot(frm, lc3d, label = '3d')
    #ax[0].plot(frm, model, label='model' )
    #print('noise:', np.std(model-lc3d), 'snr', (1-np.min(lc3d))/np.std(model-lc3d))
    ax[0].legend()
    ax[1].plot(frm, np.asarray(lc3d-lc2d), label="mean:"+str(round(np.sqrt(mn),6)))
    ax[1].fill_between(frm, np.sqrt(mn)*np.ones(len(lc3d)), -np.sqrt(mn)*np.ones(len(lc3d)), alpha=0.2)
    ax[1].set_xlabel('Phase')
    ax[1].set_ylabel('Flux')
    ax[0].set_ylabel('Flux')
    ax[0].set_title("$R_{pl}$ = 0.1 $R_{st}$, Orbit: = 2 $R_{st}$, u1: 0.6, u2:0.0, e: 0.0")
    ax[1].set_title('Residual')
    ax[1].legend()
    plt.suptitle('2D vs 3D transiting objects')
    # df = pd.DataFrame(zip(frm, lc2d, lc2dstd, lc3d, lc3dstd), columns=['frame','2d','2dstd','3d','3dstd'])
    # df.to_csv('2d3d_811.csv', index='False', sep=',')
    #np.savetxt('2d3d_0.1R_circ.csv', np.transpose(np.array([frm, lc2d, lc2dstd, lc3d, lc3dstd])),delimiter=',', header='frame, 2d, 2dstd, 3d, 3dstd')
    plt.savefig('1temp.png')
    plt.show()

#------------------------------------------------------------------------------

# df = pd.read_csv('811_fit.csv')
# ph = np.array(df['phase'])

# df_noise = df[(df.phase<-0.2) | (df.phase>0.2)]
# indf = df[(df.phase>-0.2) | (df.phase<0.2)]
# noise = np.std(np.array(df_noise['flux']))


# flux_raw = np.array(df['flux'])
# flux = fit_alg(ph, 0.334, 1.31, 1.13)
# flux3d = fit_alg_3d(ph, 0.045, 1.35,0.901)
# #off = ph[np.argmin(flux)]
# # popt2, pcov2 = curve_fit(fit_alg, indf['phase'], indf['flux'], 
# #      bounds=([0.3,1.2,1.12], [0.35,1.5,1.14]))

# plt.plot(ph, flux_raw)
# plt.plot(ph, flux)
# plt.plot(ph, flux3d)
# #plt.plot(ph, flux - flux_raw)
# plt.xlim(-0.3,0.3)

# #fluxfit=fit_alg(ph, *popt2)
# rchi=np.mean((flux-flux_raw)**2/noise**2)
# rchi3d=np.mean((flux3d-flux_raw)**2/noise**2)
# print(rchi, rchi3d)
# # plt.plot(ph, fluxfit)
# # print(popt2)
# df['model_fit']=flux
# df.to_csv('811_fit.csv', index=False)
# # print(np.mean((fluxfit-flux_raw)**2/noise**2))
# plt.savefig('temp.png')

#=======================================================================================

def shape_test(x):
    """This is for the shape testing code... here we gradually increase the sides of regular polygons and 
       and plot the LC... to see how the lightcurve changes.
    """
    global sides
    np.random.seed(3456*x)
    global fl
    sim_2d = dysim.Simulator (100, 20000, 500, fl, limb_u1=0.0, limb_u2=0.0)
    meg_2d = dysim.Megastructure(200, isrot=True)
    meg_2d.regular_polygons_2d(20,sides)
    sim_2d.add_megs(meg_2d)
    sim_2d.set_frame_length()
    if(x==0): print("Count:", meg_2d.set, x, np.pi/sim_2d.frame_length)
    sim_2d.simulate_transit()
    return(sim_2d.lc)

def multishape():
    """ continuation of the multishape program to generate multiple shapes with different number of sides.
        and comparing their lightcurves.
    """
    global fl
    global sides
    df = pd.DataFrame()

    plt.style.use('seaborn-bright')
    fig, ax = plt.subplots(1,1, figsize = (7,10), sharex=True)
    frm = np.linspace(-fl,fl, 500)
    df['frame']=frm

    for s in range(3,20):
        sides = s
    # start 4 worker processes
        with Pool(processes=40) as pool:
            lc2dsum = np.asarray(pool.map(shape_test, range(120)))
            lc2d = np.mean(lc2dsum, axis = 0)
            lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
            print("--- %s min ---" % ((time.time() - start_time)/60))

            df['sd_'+str(s)]=lc2d
            df['std_sd_'+str(s)]=lc2dstd
        
        ax.plot(frm,lc2d, label=str(s))

    with Pool(processes=40) as pool:
        lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(120)))
        lc2d = np.mean(lc2dsum, axis = 0)
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        print("--- %s min ---" % ((time.time() - start_time)/60))

        df['sd_inf']=lc2d
        df['std_sd_inf']=lc2dstd

        ax.plot(frm,lc2d, label='inf')

    ax.legend()
    ax.set_xlabel('Phase')
    ax.set_ylabel('Flux')
    plt.suptitle('Multi Shape')
    df.to_csv('multishape.csv', index='False', sep=',')
    plt.savefig('multishape.png')


#------------------------------------------------------------------

# multishape()

#======================================================================================

def solar_system(x):
    """Generating a solar system animation... not related to the previous functions
    """
    np.random.seed(x*11245)
    sim = dysim.Simulator(1, 60000, 8000, np.pi, limb_u1=0, limb_u2=0)
    meg_mc = dysim.Megastructure(83.86, True, 0.0035, ecc=0, o_vel=8.82, ph_offset=0.7)
    meg_vs = dysim.Megastructure(154.82, True, 0.0086, ecc=0, o_vel=6.44, ph_offset=0.6)
    meg_e = dysim.Megastructure(215.032, True, 0.0091, ecc=0, o_vel=5.48, ph_offset=0.5)
    meg_ms = dysim.Megastructure(326.84, True, 0.005, ecc=0, o_vel=4.42, ph_offset=0.4)
    meg_jp = dysim.Megastructure(1118.17, True, 0.102, ecc=0, o_vel=2.41, ph_offset=0.3)
    meg_st = dysim.Megastructure(2051.4, True, 0.086, ecc=0, o_vel=1.78, ph_offset=0.2)
    meg_ur = dysim.Megastructure(4128.61, True, 0.036, ecc=0, o_vel=1.25, ph_offset=0.1)
    meg_np = dysim.Megastructure(6450.9, True, 0.035, ecc=0, o_vel=1)
    sim.add_megs(meg_mc)
    sim.add_megs(meg_vs)
    sim.add_megs(meg_e)
    sim.add_megs(meg_ms)
    sim.add_megs(meg_jp)
    sim.add_megs(meg_st)
    sim.add_megs(meg_ur)
    sim.add_megs(meg_np)

    
    #if(x==0): print("Count:", meg_2d.set, x, np.pi/sim_2d.frame_length)
    sim.simulate_transit()
    return(sim.lc)

def solarsim():
    """ continulation of the solar system simulation
        values have been scaled up for visual appeal
    """
    # start 4 worker processes
    with Pool(processes=40) as pool:
        lc2dsum = np.asarray(pool.map(solar_system, range(120)))
        lc2d = np.mean(lc2dsum, axis = 0)
        lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
        print("--- %s min ---" % ((time.time() - start_time)/60))


        plt.style.use('seaborn-bright')
        fig, ax = plt.subplots(1,1, figsize = (20,7), sharex=True)

    frm = np.linspace(-np.pi,np.pi, len(lc2d))
    ax.plot(frm,lc2d,label = '2d')
    ax.legend()
    ax.set_ylabel('Flux')
    ax.set_title("$R_{pl}$ = 0.1 $R_{st}$, Orbit: = 2 $R_{st}$, u1: 0.6, u2:0.0, e: 0.0")
    plt.suptitle('2D vs 3D transiting objects')
    df = pd.DataFrame(zip(frm, lc2d, lc2dstd), columns=['frame','flux', 'std'])
    df.to_csv('solarsim.csv', index='False', sep=',')
    #np.savetxt('2d3d_0.1R_circ.csv', np.transpose(np.array([frm, lc2d, lc2dstd, lc3d, lc3dstd])),delimiter=',', header='frame, 2d, 2dstd, 3d, 3dstd')
    plt.savefig('solarsim.png')
    #plt.show()

#======================================================================================

bez_shape=[]
ca2=['#432371', '#714674', '#9F6976', '#CC8B79', '#FAAE7B']

def func(x):
    x = np.array(x)
    y=-0.0234*x**2-0.0001*x-0.0004
    return(y)

def shape_dictionary():
    np.random.seed(101010101)
    rads = np.linspace(0,1,6)
    edges = np.sqrt(1/np.linspace(0.1,0.9,5) -1)
    ns = [3,4,5,6,7,8,9,10,11,12]

    grid_size= 5
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
                for num in range(5):
                    a = get_random_points(n=n, scale=0.5) 
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
    global fl
    np.random.seed(3456*x)
    sim_2d = dysim.Simulator (1, res, 600, limb_u1=0.0, limb_u2=0.0)
    meg_2d = dysim.Megastructure(Rorb, iscircle=True, Rcircle=0.3, isrot=True)
    # meg_2d.Plcoords = np.array(bez_shape)
    sim_2d.add_megs(meg_2d)
    sim_2d.set_frame_length()
    fl = sim_2d.frame_length
    if(x==0): print("Count:", meg_2d.set)
    sim_2d.simulate_transit()
    return(sim_2d.lc, fl)

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

def analyse_scaling():
    # scaling due to rpl
    hf = h5py.File("../Shape_Directory/shape_list/n_3.hdf5", 'r')
    shape = np.array(hf.get('rad_0.6_edg_1.0_4'))
    shlist = [shape, shape*1.5, shape*2, shape*2.5, shape*3, shape*3.5]
    scale = [1,1.5,2,2.5,3,3.5]

    lclist, lcstdlist, frmlist = run_bezier_sim(shlist, 2, 10000)

    fig, ax = plt.subplots(5,1, figsize=(7,15))
    ax[0].set_title('LC for scaled shapes')
    ax[1].set_title('Residuals of scaled shapes')
    ax[2].set_title('Transit Depth vs scaling')
    ax[3].set_title('Mathematically scaled LC')
    ax[4].set_title('Mathematical scaling residuals')

    for frm,lc,sc in zip(frmlist,lclist, scale):
        ax[0].plot(frm,lc-1, label=sc)
    ax[0].legend()

    for i in range(0, len(frmlist)-1):
        ax[1].plot(frmlist[i], lclist[i+1]-lclist[i])

    minlist = [min(el)-1 for el in lclist]
    ax[2].plot([1,1.5,2,2.5,3,3.5], minlist)
    print(minlist)
    p = np.polyfit([1,1.5,2,1.5,3,3.5],minlist,2)
    #func = np.poly1d(p)

    ax[2].plot([1,1.5,2,2.5,3,3.5], func([1,1.5,2,2.5,3,3.5]))
    print(p)

    for i in range(len(frmlist)):
        dpth = func(scale[i])
        sc = dpth/min(lclist[0]-1)
        ax[3].plot(frm,sc*(lclist[0]-1))
        ax[4].plot(frm, sc*(lclist[0]-1)-lclist[i])

    plt.tight_layout()
    plt.savefig('testng_analytics.png')
    plt.show()

def analyse_scaling_rorb():
    # scaling due to rpl
    hf = h5py.File("../Shape_Directory/shape_list/n_3.hdf5", 'r')
    shape = np.array(hf.get('rad_0.6_edg_1.0_4'))
    scale = [2,4,16,32,64]
    
    lclist, lcstdlist, frmlist = run_bezier_sim([shape,shape,shape,shape,shape], scale, 20000)
    
    fig, ax = plt.subplots(4,1, figsize=(7,15))
    ax[0].set_title('LC for scaled shapes')
    ax[1].set_title('Transit Depth vs scaling')
    ax[2].set_title('Mathematically scaled LC')
    ax[3].set_title('Mathematical scaling residuals')

    for frm,lc,sc in zip(frmlist,lclist, scale):
        ax[0].plot(frm,lc-1, label=sc)
    ax[0].legend()

    #we need a phase difference list
    phlist=[]
    phlist2=[]
    for frm, lc in zip(frmlist, lclist):
        print(frm[np.where(np.abs(lc)<1)[0][0]], frm[0])
        phlist.append(np.abs(frm[np.where(np.abs(lc)<1)[0][0]]))
        phlist2.append(np.abs(frm[0]))

    ax[1].plot(np.log10(scale), np.log10(phlist))
    p = np.polyfit(np.log10(scale), np.log10(phlist),2)
    func = np.poly1d(p)
    ax[1].plot(np.log10(scale), func(np.log10(scale)))
    print(p)
    print(phlist)

    zer_lc = lclist[0][np.where(np.abs(frmlist[0])<phlist[0])[0]]
    zer_frm = np.linspace(-phlist[0],phlist[0],len(zer_lc))
    ax[2].plot(zer_frm, zer_lc, color='black')
    for frm, lc, sc, ph in zip(frmlist, lclist, scale, phlist):
        dpth = 10**(func(np.log10(sc)))
        cutlc = lc[np.where(np.abs(frm)<ph)[0]]
        cutfrm = np.linspace(-ph,ph,len(cutlc))
        scfrm = np.linspace(-dpth, dpth, len(zer_frm))
        print('check scale:',ph, dpth)
        lcnewf = interp1d(cutfrm, cutlc, fill_value='extrapolate')
        lcnew = lcnewf(scfrm)
        ax[2].plot(scfrm, zer_lc, color='blue')
        ax[2].plot(cutfrm, cutlc, color='green')
        ax[3].plot(zer_lc - lcnew)

    plt.tight_layout()
    plt.savefig('testng_analytics3.png')
    plt.show()

#--------------------------------------------------------------------------------

analyse_scaling_rorb()
#shape_dictionary()

# # accessing this
# hf = h5py.File("../Shape_Directory/shape_list/n_3.hdf5", 'r')
# for k1 in hf:
#    n = np.array(hf.get(k1))
#    print(k1)

#================================================================================