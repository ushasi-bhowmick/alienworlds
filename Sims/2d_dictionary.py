import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dysim
import time
from multiprocessing import Process, Pool

#gonna run this on hpc... will take weeks probably... hope not to drag it so far tho

#resolution setters
frame_res=50
mcmc_pts=500
frame_l=np.pi/3

#variable paramters
Rstar=100
Rpl=10
u1=0.1
u2=0.1
Rorb=1000
b=10
ecc=0
per_off=0



start_time = time.time()

def test_multi_loops_3d(x):
    np.random.seed(1234*x)
    sim_3d = dysim.Simulator (100, 50000, 500, np.pi/3, limb_u1=0.0, limb_u2=0.0)
    meg_3d = dysim.Megastructure(200, True, 20, ecc=0.0)
    sim_3d.add_megs(meg_3d)
    sim_3d.simulate_transit()
    print("Count:", meg_3d.set, x)
    return(sim_3d.lc)

def test_multi_loops_2d(x):
    global Rpl
    np.random.seed(3456*x)
    print(Rpl)
    sim_2d = dysim.Simulator (Rstar, mcmc_pts, frame_res, frame_l, limb_u1=u1, limb_u2=u2)
    meg_2d = dysim.Megastructure(Rorb, True, Rpl, isrot=True, ecc=ecc, per_off=per_off)
    sim_2d.add_megs(meg_2d)
    print("Count:", meg_2d.set, x)
    sim_2d.simulate_transit()
    return(sim_2d.lc)



if __name__ == '__main__':
    for el in [10,20,30,40]:
        Rpl=el
        # start 4 worker processes
        with Pool(processes=4) as pool:
            lc2dsum = np.asarray(pool.map(test_multi_loops_2d, range(4)))
            lc2d = np.mean(lc2dsum, axis = 0)
            lc2dstd = np.sqrt(np.mean((lc2dsum-lc2d)**2, axis=0))
            print("--- %s min ---" % ((time.time() - start_time)/60))

            plt.style.use('seaborn-bright')
            fig, ax = plt.subplots(1,1, figsize = (7,10), sharex=True)

        frm = np.linspace(-frame_l, frame_l, frame_res)
        ax.plot(frm,lc2d,label = '2d')
        ax.legend()
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')
        ax.set_title("$R_{pl}$ = 0.2 $R_{st}$, Orbit: = 2 $R_{st}$, u1: 0.0, u2:0.0, e: 0.0")
        ax.set_title('Residual')
        plt.suptitle('2D vs 3D transiting objects')
        np.savetxt('test.csv', np.transpose(np.array([frm, lc2d, lc2dstd])),delimiter=',', header='frame, 2d, 2dstd')
        plt.savefig('test'+str(el)+'.png')
    #plt.show()
