import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dy
import time
from multiprocessing import Process, Pool

start_time = time.time()

#vertical swarms: polar clark belt
#lets aim to cover the star more or less...
n = 9
Rst = 100
Rorb =200
rinp = np.pi*Rorb/(n*np.sqrt(2))
print(rinp)
sim1 = dy.Simulator(Rst, 1000, 100, np.pi)
x = np.pi/(n*np.sqrt(2))
z = np.linspace(-np.pi/2,np.pi/2, n)
print(z)

i2=0
for i in range(-int(n/2),int(n/2)+1):
    print(i*x*np.sqrt(2), np.cos(i*x), np.sin(i*x))
    meg2 = dy.Megastructure(Rorb*np.cos(i*x), False, isrot=True,
        elevation=Rorb*np.sin(i*x), ph_offset=i*x)
    meg2.Plcoords = meg2.regular_polygons_2d(rinp, 4)
    meg2.Plcoords=meg2.rotate([1,0,0],-i*x)
    meg2.Plcoords=meg2.rotate([0,1,0],i*x)
    sim1.add_megs(meg2)
    i2+=1


for i in range(-int(n/2),int(n/2)+1):

    meg2 = dy.Megastructure(Rorb*np.cos(i*x), False, isrot=True,
        elevation=Rorb*np.sin(i*x), ph_offset=np.pi-i*x)
    meg2.Plcoords = meg2.regular_polygons_2d(rinp, 4)
    meg2.Plcoords=meg2.rotate([1,0,0],i*x)
    meg2.Plcoords=meg2.rotate([0,1,0],-i*x)
    sim1.add_megs(meg2)

'''meg2 = dy.Megastructure(Rorb*np.cos(2*rinp/Rorb), False, isrot=True,
    elevation=2*rinp, ph_offset=2*rinp/Rorb)
meg2.Plcoords = meg2.regular_polygons_2d(rinp, 4)
meg2.Plcoords=meg2.rotate([1,0,0],-2*rinp/Rorb)
meg2.Plcoords=meg2.rotate([0,1,0],2*rinp/Rorb)
sim1.add_megs(meg2)

meg2 = dy.Megastructure(Rorb*np.cos(3*rinp/Rorb), False, isrot=True,
    elevation=3*rinp, ph_offset=3*rinp/Rorb)
meg2.Plcoords = meg2.regular_polygons_2d(rinp, 4)
meg2.Plcoords=meg2.rotate([1,0,0],-3*rinp/Rorb)
meg2.Plcoords=meg2.rotate([0,1,0],3*rinp/Rorb)
sim1.add_megs(meg2)'''

#print(len(sim1.megs))
sim1.simulate_transit()

print("--- %s seconds ---" % (time.time() - start_time))

TA = dy.Transit_Animate(sim1.road, sim1.megs, sim1.lc, sim1.frames)
TA.go()
#plt.plot(sim1.lc)
#plt.plot(sim2.lc)
#plt.show()