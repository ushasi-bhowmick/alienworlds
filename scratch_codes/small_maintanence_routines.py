import os
import pandas as pd
import numpy as np
from astropy.io import ascii
from astropy.table import Table

FILEPATH_IP_G="C:\\Users\\ushas\\Documents\\nonpl_fps_red2\\global\\"
FILEPATH_IP_L="C:\\Users\\ushas\\Documents\\nonpl_fps_red2\\local\\"
FILEPATH_OP="C:\\Users\\ushas\\Documents\\trash_rebins\\npl_fps\\"
FILEPATH_RAW_PL="F:\Masters_Project_Data\\alienworlds_data\\"
FILEPATH_RAW_FPS="F:\Masters_Project_Data\\alienworlds_fps\\"
FILEPATH_RAW_OTH="F:\Masters_Project_Data\\alienworlds_others\\"

def shift_baddata_elsewhere_g():
    entries_g=os.scandir(FILEPATH_IP_G)
    for el in entries_g:
        df=pd.read_csv(FILEPATH_IP_G+el.name)
        if(len(df)>2000):
            print(len(df))
            #print(FILEPATH_IP_L+el.name[0:11]+"_l "+FILEPATH_OP)
            os.system("move "+FILEPATH_IP_G+el.name+" "+FILEPATH_OP)
            #os.system("move "+FILEPATH_IP_L+el.name[0:11]+"_l "+FILEPATH_OP)

def trim_off_longer_samples():
    entries=os.scandir(FILEPATH_OP)
    for el in entries:
        df=pd.read_csv(FILEPATH_OP+el.name,sep=" ")
        y=df.tail(-1)
        y.to_csv(FILEPATH_IP_G+el.name,sep=' ',index=False)
        print(len(y))

def move_doubtful_files_elsewhere():
    ref=ascii.read("ex_koi_id.tab")
    koi_kepid=[('00000'+str(el))[-9:] for el in ref['kepid']]
    direct=os.listdir(FILEPATH_RAW_FPS)
    dir_kepid=[el[4:13] for el in direct]

    for stuff in direct:
        try:
            ind=koi_kepid.index(stuff[4:13])
            if(ref['koi_disposition'][ind] == 'CONFIRMED'):
                print(stuff,"move "+FILEPATH_RAW_FPS+stuff+" "+FILEPATH_RAW_PL)
                os.system("move "+FILEPATH_RAW_FPS+stuff+" "+FILEPATH_RAW_PL)
        except:
            print("not:", stuff)
            os.system("move "+FILEPATH_RAW_FPS+stuff+" "+FILEPATH_RAW_OTH)
    
def get_proper_labels_from_koi_table():
    ref=ascii.read("ex_koi_id.tab")
    koi_kepid=[('00000'+str(el))[-9:] for el in ref['kepid']]
    koi_label=ref['koi_disposition']
    tce_num=[]
    tce_num.append(1)
    for i in range(1,len(koi_kepid)):
        if(koi_kepid[i]==koi_kepid[i-1]):
            tce_num.append(tce_num[-1]+1)
        else: tce_num.append(1)
    data=Table()
    print(len(koi_kepid),len(tce_num),len(koi_label))
    data['kepid']=koi_kepid
    data['tce_plnt_num']=tce_num
    data['label']=koi_label
    ascii.write(data,"robovetter_label.dat",overwrite=True)

def check_new_binary():
    eb_entry=ascii.read('eb_label.dat')
    av_entry=ascii.read('robovetter_label.dat')
    kep_eb=[('00000'+str(el))[-9:] for el in eb_entry['kepid']]
    kep_av=[('00000'+str(el))[-9:] for el in av_entry['kepid']]
    print(kep_eb[0:10],kep_av[0:10])
    i=0
    for el in kep_eb:
        print(np.where(np.array(kep_av)==el)[0],i)
        i+=1

def reorder_TS():
    X_train=np.loadtxt('training_data/Xtrain_rv_raw500_3.csv',delimiter=',')
    Y_train=np.loadtxt('training_data/Ytrain_rv_raw500_3.csv',delimiter=',') 
    nY_train=[]
    for el in Y_train:
        if(el[0]==1): nY_train.append([1,0])
        else: nY_train.append([0,1])

    temp1=[]
    temp2=[]
    filtind=[i for i in range(0,len(nY_train)) if (nY_train[i]==np.array([1,0])).all()]
    filtind2=[i for i in range(0,len(nY_train)) if (nY_train[i]==np.array([0,1])).all()]
    print(len(filtind),len(filtind2)) 
    #print(min(len(filtind[0]),len(filtind2[0])))
    for i in range(0,min(len(filtind),len(filtind2))):
        temp1.append(X_train[filtind[i]])
        temp2.append(nY_train[filtind[i]])
        temp1.append(X_train[filtind2[i]])
        temp2.append(nY_train[filtind2[i]])
    print(np.array(temp1).shape,np.array(temp2).shape)
    np.savetxt('training_data/Xtrain_rv_raw500_2_temp.csv', np.array(temp1), delimiter=',')
    np.savetxt('training_data/Ytrain_rv_raw500_2_temp.csv', np.array(temp2), delimiter=',')

reorder_TS()
        
