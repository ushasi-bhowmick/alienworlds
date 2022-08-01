import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d
from astropy.io import fits,ascii
import tensorflow as tf
import GetLightcurves as gc
import h5py

""" BEZIER CURVE SIMULATIONS WITH MACHINE LEARNING
This module is to make a training sample for training a machine learning algorithm that can figure out the
shape of a transiting object from the Lightcurve (don't know if it will work but lets see). Here we build up 
a TFR records file from the directory... coz thats kinda an ML thing.

"""

def tfr_from_bezier(maxres, pixelres, trainfrac, trainfile, testfile):
    entries = os.listdir("../Shape_Directory/shape_lc/")
    cumsuminp=[]
    cumsumop=[]
    # maxres = 420
    i = 0 
    for entry in entries:
        hf = h5py.File("../Shape_Directory/shape_lc/"+entry, 'r')
        for key in hf:
            if(key.find('lc')>0):
                i+=1
                cumsuminp.append(hf.get(key[:-2]+'lc'))
                shape = hf.get(key[:-2]+'sh')
                x = np.linspace(0,1,len(shape))
                intx = interp1d(x, shape[:,0])
                inty = interp1d(x, shape[:,1])

                newx = np.linspace(0,1, maxres)
                newshx = intx(newx)
                newshy = inty(newx)
                cumsumop.append([newshx, newshy])

                print('count', i, len(newshx), len(newshy))
    
    cumsuminp = np.array(cumsuminp)
    cumsumop = np.array(cumsumop)
    print('Total Sample:', cumsuminp.shape, cumsumop.shape)

    cumsumpix=[]
    
    grid = np.array([i*10 for i in np.linspace(-1,1,int(pixelres)+1)])
    print('Grid:',grid)
    for el in cumsumop:
        pixels = np.zeros((int(pixelres), int(pixelres)))
        for x, y  in zip(el[0], el[1]):
            inx = [i for i in range(len(grid)-1) if(x*10>grid[i] and x*10<grid[i+1])][0]
            iny = [i for i in range(len(grid)-1) if(y*10>grid[i] and y*10<grid[i+1])][0]
            pixels[inx][iny]=1
        cumsumpix.append(pixels)

    #shuffle the data so that its random
    mixarr = np.arange(0,len(cumsuminp),1)
    np.random.shuffle(mixarr)
    cap = int(len(mixarr)*trainfrac)
    trainX = [cumsuminp[i] for i in mixarr[0:cap]]
    trainY = [cumsumop[i] for i in mixarr[0:cap]]
    trainPix = [cumsumpix[i] for i in mixarr[0:cap]]
    testX = [cumsuminp[i] for i in mixarr[cap:]]
    testY =[cumsumop[i] for i in mixarr[cap:]]
    testPix = [cumsumpix[i] for i in mixarr[cap:]]

    netTrain = [[x,y,z] for x,y,z in zip(trainX, trainY, trainPix)]
    netTest = [[x,y,z] for x,y,z in zip(testX, testY, testPix)]

    print(np.array(netTrain).shape, np.array(netTest).shape)
    gc.write_tfr_record('../Sims/bezier_train',np.array(netTrain),['lc','shape','pixel'],['ar','ar','ar'],['float32','float32','int8'])
    gc.write_tfr_record('../Sims/bezier_test',np.array(netTest),['lc','shape','pixel'],['ar','ar','ar'],['float32','float32','int8'])



tfr_from_bezier(420, 50, 0.8, 'some', 'some')
