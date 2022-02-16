import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import aliensims as dysim
import time



start_time = time.time()

sim_2d = dysim.Simulator (100, 1000, 400, np.pi/2)
sim_3d = dysim.Simulator (100, 1000, 400, np.pi/2)

th = np.linspace(0, 2*np.pi, 120)
Plcoord = np.array([[10*np.cos(el), 10*np.sin(el), 0] for el in th])

meg_2d = dysim.Megastructure(200, False, Plcoords=1*Plcoord, isrot=True)
meg_3d = dysim.Megastructure(200, True, 10)
sim_2d.add_megs(meg_2d)
sim_3d.add_megs(meg_3d)


sumlc2d = []
sumlc3d = []
#start up the simulation and average out
for i in range(0,2):
    sim_2d.simulate_transit()
    sim_3d.simulate_transit()
    sumlc2d.append(sim_2d.lc)
    sumlc3d.append(sim_3d.lc)
    sim_2d.reinitialize()
    sim_3d.reinitialize()

lc2d = np.mean(np.array(sumlc2d),axis =0)
lc3d = np.mean(np.array(sumlc3d), axis=0)

print("--- %s seconds ---" % (time.time() - start_time))

mn = (np.asarray(lc3d-lc2d)**2).sum()/len(lc3d)
print(np.sqrt(mn))

plt.style.use('seaborn-bright')
fig, ax = plt.subplots(2,1, figsize = (7,10), sharex=True)

ax[0].plot(sim_2d.frames,lc2d,label = '2d')
ax[0].plot(sim_3d.frames, lc3d, label = '3d')
ax[0].legend()
ax[1].plot(sim_2d.frames, np.asarray(lc3d-lc2d), label="mean:"+str(round(np.sqrt(mn),6)))
ax[1].fill_between(sim_2d.frames, np.sqrt(mn)*np.ones(len(lc3d)), -np.sqrt(mn)*np.ones(len(lc3d)), alpha=0.2)
ax[1].set_xlabel('Phase')
ax[1].set_ylabel('Flux')
ax[0].set_ylabel('Flux')
ax[0].set_title("$R_{pl}$ = 0.1 $R_{st}$, Orbit: 200")
ax[1].set_title('Residual')
ax[1].legend()
plt.suptitle('2D vs 3D transiting objects')
#np.savetxt('2d3d_0.1R_sm_4.csv', np.transpose(np.array([sim_2d.frames, lc2d, lc3d])),delimiter=' ', header='frame, 2d, 3d')
#plt.savefig('2d3d_res_0.1R_sm_4.png')
plt.show()

#add up all th input bits and make wholesome lc
'''set1 = np.loadtxt('2d3d_0.1R_sm.csv', delimiter=' ' )
set2 = np.loadtxt('2d3d_0.1R_sm_2.csv', delimiter=' ' )
set3 = np.loadtxt('2d3d_0.1R_sm_3.csv', delimiter=' ' )
set4 = np.loadtxt('2d3d_0.1R_sm_4.csv', delimiter=' ' )
#set5 = np.loadtxt('2d3d_0.05R_sm_5.csv', delimiter=' ' )

fr = set1[:,0]
lc2d1 = set1[:,1]
lc2d2 = set2[:,1]
lc2d3 = set3[:,1]
lc2d4 = set4[:,1]
#lc2d5 = set5[:,1]
lc3d1 = set1[:,2]
lc3d2 = set2[:,2]
lc3d3 = set3[:,2]
lc3d4 = set4[:,2]
#lc3d5 = set5[:,2]

print(set1.shape)

sumlc2d = np.mean(np.array([lc2d1,lc2d2,lc2d3,lc2d4]), axis = 0)
sumlc3d = np.mean(np.array([lc3d1,lc3d2,lc3d3,lc3d4]), axis = 0)

mn = (np.asarray(sumlc3d-sumlc2d)**2).sum()/len(sumlc3d)
print(np.sqrt(mn))

plt.style.use('seaborn-bright')
fig, ax = plt.subplots(2,1, figsize = (7,10), sharex=True)

ax[0].plot(fr, sumlc2d,label = '2d')
ax[0].plot(fr, sumlc3d, label = '3d')
ax[0].legend()
ax[1].plot(fr, np.asarray(sumlc3d-sumlc2d), label="mean:"+str(round(np.sqrt(mn),6)))
ax[1].fill_between(fr, np.sqrt(mn)*np.ones(len(sumlc3d)), -np.sqrt(mn)*np.ones(len(sumlc3d)), alpha=0.2)
ax[1].set_xlabel('Phase')
ax[1].set_ylabel('Flux')
ax[0].set_ylabel('Flux')
ax[0].set_title("$R_{pl}$ = 0.1 $R_{st}$, Orbit: 200")
ax[1].set_title('Residual')
ax[1].legend()
plt.suptitle('2D vs 3D transiting objects')
plt.savefig('2d3d_res_0.1R_sm_main.png')


res1 = lc3d1[np.where(lc3d1<1)[0]] - lc2d1[np.where(lc3d1<1)[0]]
res2 = lc3d2[np.where(lc3d2<1)[0]] - lc2d2[np.where(lc3d2<1)[0]]
res3 = lc3d3[np.where(lc3d3<1)[0]] - lc2d3[np.where(lc3d3<1)[0]]
res4 = lc3d4[np.where(lc3d4<1)[0]] - lc2d4[np.where(lc3d4<1)[0]]
#res5 = lc3d5[np.where(lc3d5<1)[0]] - lc2d5[np.where(lc3d5<1)[0]]
resav = sumlc3d[np.where(sumlc3d<1)[0]] - sumlc2d[np.where(sumlc3d<1)[0]]
print(len(res1))

fig2, ax2 = plt.subplots(3,2, figsize = (10,10), sharex=True)
plt.suptitle('Residual Distribution Histograms')
ax2[0][0].hist(res1*10000,20, alpha=0.5)
ax2[1][0].hist(res2*10000,20, alpha=0.5)
ax2[0][1].hist(res3*10000,20, alpha=0.5)
ax2[1][1].hist(res4*10000,20, alpha=0.5)
#ax2[2][0].hist(res5*10000,20, alpha=0.5)
ax2[2][1].hist(resav*10000,20)
ax2[2][1].set_xlabel('Residual($10^{-4}$)')
ax2[2][0].set_xlabel('Residual($10^{-4}$)')
ax2[2][1].set_title('Average')
plt.savefig('2d3d_res_0.1R_sm_main_res.png')
plt.show()'''