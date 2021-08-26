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

get_proper_labels_from_koi_table()
        
