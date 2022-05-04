from cmath import phase
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, interpn
from transit import occultnonlin, occultquad
from scipy import interpolate
import time

#we need to change some strategies, because its too slow
#Im gonna try scipy

def test_func(p,u1,u2,rorb,imp):
    #read all the data and store it in dataframes ... ig?
    df_master=[]
    rpl_arr=np.around(np.linspace(0.01,0.5, 10) ,2)
    rorb_arr=np.around(np.logspace(0.31,3,10), 2)
    b_arr=[0.0,0.2,0.4,0.6,0.8]
    u1_arr=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    u2_arr=[0.0,0.2,0.4]

    #now we make the grid:
    points=(rpl_arr, rorb_arr, b_arr, u1_arr, u2_arr)

    vals=[]
    phases=[]
    for rs in rpl_arr:
        net_r=[]
        net_rph=[]
        for ros in rorb_arr:
            df=pd.read_csv('../Computation_Directory/Rpl_'+str(np.around(rs*100,2))+'/2d_rorb_'+str(ros)+'.csv')

            print(rs, ros)
            op = [[[np.array(df['u1_'+str(u1s)+'_u2_'+str(u2s)+'_b_'+str(bs)]) for u2s in u2_arr] 
            for u1s in u1_arr] for bs in b_arr]
            
            net_r.append(op)
            net_rph.append(np.array(df['frame']))
        vals.append(net_r)
        phases.append(net_rph)


    #grid = np.meshgrid(*points)

    print(np.array(vals).shape, np.array(phases).shape)
    




def new_plar(ph,p,u1,u2,rorb,imp):
    znp = np.sqrt(np.abs(rorb*np.sin(ph*np.pi))**2+imp**2)
    a= occultquad(znp,p,[u1,u2])  
    return(a)

#five point linear interpolation
def formula(val, parspace, rorb, rpl, b, u1, u2):
    #val is an array of 32, sgima is an array of 32*5 and slp is an array of 5
    #output is one point
    
    val2=[interp1d(parspace[4][:2], [val[i],val[i+1]],kind='linear', fill_value='extrapolate')(u2) for i in range(0,len(val),2)]
    val2=[interp1d(parspace[3], [val2[i],val2[i+1],val2[i+2]], kind='quadratic', fill_value='extrapolate')(u1) for i in range(0,len(val2)-2,3)]
    val2=[interp1d(parspace[2][:2], [val2[i],val2[i+1]], kind='linear', fill_value='extrapolate')(b) for i in range(0,len(val2),2)]
    val2=[interp1d(parspace[1], [val2[i],val2[i+1],val2[i+2]], kind='quadratic', fill_value='extrapolate')(rpl) for i in range(0,len(val2)-2,3)]
    val2=[interp1d(parspace[0], [val2[i],val2[i+1],val2[i+2]], kind='quadratic', fill_value='extrapolate')(rorb) for i in range(0,len(val2),3)]
    
    return(val2)

def lc_read(rpl, rorb, b, u1, u2):

    rpl_arr=np.around(np.linspace(0.01,0.5, 10) ,2)
    #rpl_arr=[rpl_arr[i] for i in range(0,len(rpl_arr)) if i!=2]
    rorb_arr=np.around(np.logspace(0.31,3,10), 2)
    #rorb_arr=[rorb_arr[i] for i in range(0,len(rorb_arr)) if i!=8]
    b_arr=[0,0.2,0.4,0.6,0.8]
    u1_arr=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    u2_arr=[0.0,0.2,0.4]
    #print(rorb_arr, rpl_arr)

    listed=[]
    parspace=[]
    vs=[[rorb_arr[i],rorb_arr[i+1],rorb_arr[i+2]] for i in range(len(rorb_arr)-2) if(rorb<rorb_arr[i+2] and rorb>rorb_arr[i])]
    if(rorb<rorb_arr[0]): vs=[[rorb_arr[0],rorb_arr[1],rorb_arr[2]]]
    if(rorb>rorb_arr[-1]): vs=[[rorb_arr[-3],rorb_arr[-2],rorb_arr[-1]]]
    parspace.append(vs[0])
    vs=vs[0]
    listed.append(np.concatenate((np.repeat(vs[0],36),np.repeat(vs[1],36),np.repeat(vs[2],36)) ))
    
    vs=[[rpl_arr[i],rpl_arr[i+1],rpl_arr[i+2]] for i in range(0,len(rpl_arr)-2) if(rpl<rpl_arr[i+2] and rpl>rpl_arr[i])]
    if(rpl<rpl_arr[0]): vs=[[rpl_arr[0],rpl_arr[1],rpl_arr[2]]]
    if(rpl>rpl_arr[-1]): vs=[[rpl_arr[-3],rpl_arr[-2],rpl_arr[-1]]]
    parspace.append(vs[0])
    vs=vs[0]
    temp=np.concatenate((np.repeat(vs[0],12),np.repeat(vs[1],12),np.repeat(vs[2],12)))
    listed.append(np.tile(temp,3))
    
    vs=[[b_arr[i],b_arr[i+1],0] for i in range(len(b_arr)-1) if(b<b_arr[i+1] and b>b_arr[i])]
    if(b<b_arr[0]): vs=[[b_arr[0],b_arr[1],0]]
    if(b>b_arr[-1]): vs=[[b_arr[-3],b_arr[-2],0]]
    parspace.append(vs[0])
    vs=vs[0]
    temp=np.concatenate((np.repeat(vs[0],6),np.repeat(vs[1],6)))
    listed.append(np.tile(temp,9))
    
    vs=[[u1_arr[i],u1_arr[i+1],u1_arr[i+2]] for i in range(len(u1_arr)-2) if(u1<u1_arr[i+2] and u1>u1_arr[i])]
    if(u1<u1_arr[0]): vs=[[u1_arr[0],u1_arr[1],u1_arr[2]]]
    if(u1>u1_arr[-1]): vs=[[u1_arr[-3],u1_arr[-2],u1_arr[-1]]]
    parspace.append(vs[0])
    vs=vs[0]
    temp=np.concatenate((np.repeat(vs[0],2),np.repeat(vs[1],2),np.repeat(vs[2],2)))
    listed.append(np.tile(temp,18))
    
    vs=[[u2_arr[i],u2_arr[i+1],0] for i in range(len(u2_arr)-1) if(u2<u2_arr[i+1] and u2>u2_arr[i])]
    if(u2<u2_arr[0]): vs=[[u2_arr[0],u2_arr[1],0]]
    if(u2>u2_arr[-1]): vs=[[u2_arr[-3],u2_arr[-2],0]]
    parspace.append(vs[0])
    vs=vs[0]
    listed.append(np.tile([vs[0],vs[1]],54))
    

    listed = np.transpose(np.array(listed))
    
    phase_list=[]
    flux_list=[]
    for el in listed:
        df=pd.read_csv('../Computation_Directory/Rpl_'+str(el[1]*100)+'/2d_rorb_'+str(el[0])+'.csv')
        ph=df['frame']
        fl=df['u1_'+str(el[3])+'_u2_'+str(el[4])+'_b_'+str(el[2])]
        phase_list.append(np.array(ph))
        flux_list.append(np.array(fl))

    return(phase_list, flux_list, parspace)

def lc_interpolate(ph, rpl, rorb, b, u1, u2):
    phase_list, flux_list, parspace = lc_read(rpl, rorb, b, u1, u2)
    
    
    final_flux= np.transpose(flux_list)

    #print(parspace)
    op_flux=[]
    for fl in final_flux:
        op_flux.append(formula(fl,parspace, rorb, rpl, b, u1, u2))

    #phase interpolation
    #print([phase_list[0][-1]/np.pi,phase_list[36][-1]/np.pi,phase_list[-1][-1]/np.pi])
    # phf=interp1d(parspace[0], [phase_list[0][-1],phase_list[54][-1],phase_list[-1][-1]], kind='quadratic', fill_value='extrapolate')(rorb)
    phf=interp1d(parspace[0], np.log10([phase_list[0][-1],phase_list[54][-1],phase_list[-1][-1]]),
       kind='linear', fill_value='extrapolate')(rorb)
    phf=10**(phf)
    
    ph_final=np.linspace(-phf,phf,300)
    op_flux = np.array(op_flux)
    f = interp1d(ph_final,op_flux[:,0],kind='linear', fill_value='extrapolate')

    return(f(ph))

tic=time.time()

# def value_func_3d(x, y, z):
#     return 2 * x + 3 * y - z
# x = np.linspace(0, 4, 5)
# y = np.linspace(0, 5, 6)
# z = np.linspace(0, 6, 7)
# points = (x, y, z)
# values = value_func_3d(*np.meshgrid(*points, indexing='ij'))
# point = np.array([2.21, 3.12, 1.15])
# print(interpn(points, values, point))

# print( np.array(values).shape)
test_func(1,1,1,1,1)

print("time:",time.time()-tic," s")

# a,b, t1, t2=lc_interpolate(0.12,502.47,0.201,0.3,0.201)
# print(len(a),np.array(b).shape)

# fig, ax = plt.subplots(2,1,figsize=(7,10))
# ax[0].plot(a,np.array(b),label='interpolation')

# #plt.plot(a,np.mean(b,axis=1))
# # for x,y in zip(t1,t2):
# #     plt.plot(x,y,color='black')

# df=pd.read_csv('../Computation_Directory/Rpl_'+str(12.0)+'/2d_rorb_'+str(502.47)+'.csv')
# #df=pd.read_csv('2d3d_0.1R_limb_circ.csv')
# ph=df['frame']
# #fl=df['u1_'+str(0.3)+'_u2_'+str(0.2)+'_b_'+str(0.2)]
# fl=df['u1_0.3_u2_0.2_b_0.2']
# #fl=np.array(df['2d'])
# func=interp1d(ph, fl, kind='linear', fill_value='extrapolate')
# b2 = func(a)
# ax[0].plot(ph,fl, color='red', label='true')
# ax[0].legend()
# ax[1].plot(a,np.array(b)[:,0]-b2)

# ax[0].set_title('Rpl:0.12, Rorb:502.47, u1:0.3, u2:0.2, b:0.2')
# #model = new_plar(np.array(ph)/np.pi,0.12, 0.3,0.2,2.04,0.2)

# #plt.plot(ph, model)
# plt.savefig('interpset3.jpg')
# plt.show()
# ph = np.linspace(-np.pi/2, np.pi/2, 2000)
# lc = lc_interpolate(ph,0.12,2.47,0.201,0.3,0.201)
# plt.plot(ph, lc)
# plt.show()