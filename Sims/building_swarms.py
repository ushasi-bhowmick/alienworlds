import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dy
import time
from multiprocessing import Process, Pool

#vertical swarms: polar clark belt
n = 11
Rst = 100
Rorb =200
rinp = np.pi*Rorb/(n*np.sqrt(2))
print(rinp)
sim1 = dy.Simulator(Rst, 1000, 100, np.pi/2)

for i in range(int(n/2)+1):
    meg = dy.Megastructure(Rorb*np.cos(i*np.pi/n), False, isrot=True, elevation=Rorb*np.sin(i*np.pi/n))
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],-i*np.pi/n)
    sim1.add_megs(meg)

for i in range(1,int(n/2)+1):
    meg = dy.Megastructure(Rorb*np.cos(i*np.pi/n), False, isrot=True, elevation=-Rorb*np.sin(i*np.pi/n))
    meg.Plcoords = meg.regular_polygons_2d(rinp, 4)
    meg.Plcoords=meg.rotate([0,0,1],np.pi/4)
    meg.Plcoords=meg.rotate([1,0,0],i*np.pi/n)
    sim1.add_megs(meg)

sim1.simulate_transit()

#
TA = dy.Transit_Animate(sim1.road, sim1.megs, sim1.lc, sim1.frames)
TA.go()
#plt.plot(sim1.lc)
#plt.plot(sim2.lc)
#plt.show()