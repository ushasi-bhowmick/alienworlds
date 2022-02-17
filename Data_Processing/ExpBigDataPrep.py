import numpy as np
from astropy.io import ascii, fits
import pandas as pd
import os
import tensorflow as tf
import GetLightcurves as gc


CATALOG="../../Catalogs/"
TRAINING_MODULE="../../processed_directories/full_lc_residuals/"

#this is my personal devised NN ... may or may not work but I have some hopes up from it.

#this function takes out a chunk from the LC... coz the whole thing may be too computationally expensive to deal with.
#the chunk is 1/3 of the whole thing that amounts to about 24000 input samples... should be enough to run a conv or autoencoder NN
#we check the transit periods and sort the data into good ones and problematic ones. The good ones are those where the selected chunk most 
#definitely has a transit... te problematic ones may need manual intervention to select targets.

#we need to do something about the NaN values better to interpolate between the nan values
def take_out_from_dv():
    FILEPATH_RAW_PL="F:\Masters_Project_Data\\alienworlds_data\\"
    FILEPATH_RAW_FPS="F:\Masters_Project_Data\\alienworlds_fps\\"
    FILEPATH_RAW_OTH="F:\Masters_Project_Data\\alienworlds_others\\"
    entry_pl=os.scandir(FILEPATH_RAW_PL)
    entry_fps=os.scandir(FILEPATH_RAW_FPS)

    for el in entry_pl:
        hdu=fits.open(FILEPATH_RAW_PL+el.name)
        num=len(hdu)
        obsdur=hdu[1].data["TIME"][-1]-hdu[1].data["TIME"][0]
        dang_list=[hdu[i].header['NTRANS']*hdu[i].header['TPERIOD'] for i in range(1,num-1)]
        datl=len(hdu[1].data["TIME"])

        x=hdu[1].data["TIME"][int(datl*7/15):int(datl*8/15)]
        y=hdu[1].data["LC_WHITE"][int(datl*7/15):int(datl*8/15)]
        yr=hdu[1].data["LC_WHITE"][int(datl*7/15):int(datl*8/15)]

        if(np.any(dang_list<obsdur/2)): 
            print(el.name[4:13],dang_list,obsdur/2)
            df=pd.DataFrame(list(zip(x, y)),columns =['time', 'flux'])
            df.to_csv("ushasi_NN_data/short_prob_ones/"+el.name[4:13],sep=' ',index=False)
            continue

        nancount=np.isnan(y).sum()
        if(nancount/len(y)>0.2): 
            print("nans:",nancount/len(y))
            df=pd.DataFrame(list(zip(x, y)),columns =['time', 'flux'])
            df.to_csv("ushasi_NN_data/short_prob_ones/"+el.name[4:13],sep=' ',index=False)
            continue

        for i in range(0,len(y)):
            if(np.isnan(y[i])):
                t=1
                try:
                    while(np.isnan(y[i-t]) or np.isnan(y[i+t])):    
                        if(i-t <0):
                            y[i]=y[i+t]
                            break
                        if(i+t >len(x)): 
                            y[i]=y[i-t]
                            break
                        t+=1
                    y[i]=(y[i-t]+y[i+t])/2
                except:
                    y[i]=0

        print("clean",len(x),np.isnan(y).sum(),np.isnan(yr).sum())
        df=pd.DataFrame(list(zip(x, y)),columns =['time', 'flux'])
        #print(el.name[4:13],dang_list,obsdur/3,len(df))
        df.to_csv("ushasi_NN_data/short_good_ones/"+el.name[4:13],sep=' ',index=False)   
        

#now the next question is assigning labels and making a working NN training sample out of this. To do this we use the filtered
#robovetter catalog... Now some TCE targets will inadvertently invite planet candidates into them... we'll label them as planets
#coz we can't create another class out of it
def get_labels_and_make_TS():
    FILEPATH="ushasi_NN_data/short_good_ones/"
    entry=os.listdir(FILEPATH)
    cat=ascii.read('robovetter_label.dat')
    kepid=cat['kepid']
    kepid=[('00000'+str(el))[-9:] for el in kepid]
    label=cat['label']
    Xtrain=[]
    Ytrain=[]

    for el in entry:
        df=pd.read_csv(FILEPATH+el,sep=" ")
        loc=np.where(np.array(kepid)==el)
        this_label=np.array([label[m] for m in loc[0]])
        if(np.all(this_label=='CANDIDATE')): continue
        temparr=[0,0]
        #if(np.any(this_label=='CONFIRMED') or np.any(this_label=='CANDIDATE')): temparr[0]=1
        #if(np.any(this_label=='FPS')): temparr[1]=1
        if(this_label[0]=='CONFIRMED' or this_label[0]=='CANDIDATE'): temparr[0]=1
        if(this_label[0]=='FPS'): temparr[1]=1

        if(np.all(temparr==1)): continue
        print(len(df),temparr,this_label)
        Xtrain.append(np.array(df['flux']))
        Ytrain.append(np.array(temparr))


    np.savetxt('training_data/Xtrain_ush_short_nolap.csv', np.transpose(np.array(Xtrain)), delimiter=',')
    np.savetxt('training_data/Ytrain_ush_short_nolap.csv', np.array(Ytrain), delimiter=',')

def cumulative_ts(raw_path, outpath):
    #phf_entry = os.listdir(phfold_path)
    print(raw_path)
    raw_entry = os.listdir(raw_path)

    oPGtest, oPLtest, oLtest, oeltest, oPGval, oPLval, oLval, oelval, oPGtrain, oPLtrain, oLtrain, oeltrain = check_tfr_records()
    
    Xtrain=[]
    Mtrain=[]
    PLtrain=[]
    PGtrain=[]
    Ctrain=[]
    Ltrain=[]
    eltrain=[]

    Xval=[]
    Mval=[]
    PLval=[]
    PGval=[]
    Cval=[]
    Lval=[]
    elval=[]

    Xtest=[]
    Mtest=[]
    PLtest=[]
    PGtest=[]
    Ctest=[]
    Ltest=[]
    eltest=[]

    note=0
    for el in range(len(oeltrain)):
        note+=1
        print(oeltrain[el])
        try: loc = np.where(np.asarray(raw_entry)==str(oeltrain[el]))[0]
        except: 
            print("not found")
            continue
        if(len(loc)==0):
            print("not found")
            continue

        tempx=[]
        tempm=[]
        tempc=[]

        #nums = np.asarray(av_no[loc])

        dfr_i,dfr_m,dfr_c = gc.read_tfr_record(raw_path+oeltrain[el],['input','mask','counts'],
            ['ar','ar','ar'], [tf.float32, tf.bool, tf.int8])

        dfr_i = np.asarray(dfr_i)
        dfr_m = np.asarray(dfr_m)
        dfr_c = np.asarray(dfr_c)

        els,cts = np.unique(dfr_c,axis=0,return_counts=True)
        bestarr = els[np.argmax(cts)]
        best_inds = [i for i in range(0,len(dfr_c)) if(np.all(dfr_c[i]==np.asarray(bestarr)))]
        np.random.shuffle(best_inds)

        try: temp=dfr_m.reshape(len(dfr_c),4000,3)
        except: continue

        mintab = int(min(1,len(best_inds)))
        for inds in best_inds[mintab:]:
            if(len(dfr_i[inds])<4000): 
                print('odd')
                continue
            check_noise=[(temp[inds,i]==np.asarray([0,0,1])).all() for i in range(0,len(temp[inds]))]
            if(np.asarray(check_noise).all()): 
                print('noise',oeltrain[el])
                continue
            
            tempx.append(dfr_i[inds])
            tempm.append(dfr_m[inds])
            tempc.append(dfr_c[inds])
            break
        if(len(tempx)<1): 
            print('miss')
            continue
        note+=1
         
        Xtrain.append(tempx[0])
        Mtrain.append(tempm[0])
        Ctrain.append(tempc[0])
        PLtrain.append(oPLtrain[el])
        PGtrain.append(oPGtrain[el])
        eltrain.append(oeltrain[el])
        Ltrain.append(oLtrain[el])
 
        print(note,tempx[0].shape,tempm[0].shape,tempc[0], oLtrain[el])

    Mtrain=np.asarray(Mtrain, np.bool)
    Xtrain=np.asarray(Xtrain, np.float32)
    Ctrain=np.asarray(Ctrain, np.int8)
    PLtrain=np.asarray(PLtrain, np.float32)
    PGtrain=np.asarray(PGtrain, np.float32)
    Ltrain=np.asarray(Ltrain, np.bool)
    eltrain = np.asarray(eltrain)
  
    print("stats: ",np.asarray(Xtrain).shape, np.asarray(Mtrain).shape, np.asarray(PLtrain).shape,
        np.asarray(PGtrain).shape, np.asarray(Ltrain).shape, np.asarray(eltrain).shape)
    
    net = np.asarray([[Xtrain[i],Mtrain[i],PLtrain[i],PGtrain[i],Ctrain[i],
        Ltrain[i],eltrain[i]] for i in range(len(Ltrain))], dtype='object')
    gc.write_tfr_record(outpath+'test',net,['input','map','local','global','counts','label','id'],
        ['ar','ar','ar','ar','ar','ar','b'],['float32','bool','float32','float32','int8','bool','byte'])

def _parse_function(example_proto):
  feature_description = {
        'global_view': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
        'local_view': tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0,allow_missing=True),
        'av_training_set': tf.io.FixedLenFeature([], tf.string, default_value=''), 
        'kepid': tf.io.FixedLenFeature([], tf.int64, default_value=0000), 
    }
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def one_hot(val):
    val=val.decode("utf-8") 
    if(val=='AFP'): return([0,1])
    elif(val=='PC'):  return([1,0])
    elif(val=='NTP'): return([0,1])

def check_tfr_records():
    TRAIN_MOD = '../../training_data/'
    Xtrain=[]
    Ytrain=[]
    Xtrainl=[]
    Xval=[]
    Yval=[]
    Xvall=[]
    Xtest=[]
    Ytest=[]
    Xtestl=[]
    Xid=[]
    Xvalid=[]
    Xtestid=[]

    raw_data1=tf.data.TFRecordDataset([TRAIN_MOD+'train-00000-of-00008'])
    raw_data2=tf.data.TFRecordDataset([TRAIN_MOD+'train-00001-of-00008'])
    raw_data3=tf.data.TFRecordDataset([TRAIN_MOD+'train-00002-of-00008'])
    raw_data4=tf.data.TFRecordDataset([TRAIN_MOD+'train-00003-of-00008'])
    raw_data5=tf.data.TFRecordDataset([TRAIN_MOD+'train-00004-of-00008'])
    raw_data6=tf.data.TFRecordDataset([TRAIN_MOD+'train-00005-of-00008'])
    raw_data7=tf.data.TFRecordDataset([TRAIN_MOD+'train-00006-of-00008'])
    raw_data8=tf.data.TFRecordDataset([TRAIN_MOD+'train-00007-of-00008'])
    raw_val=tf.data.TFRecordDataset([TRAIN_MOD+'val-00000-of-00001'])
    raw_test=tf.data.TFRecordDataset([TRAIN_MOD+'test-00000-of-00001'])

    parsed_dataset1 = raw_data1.map(_parse_function)
    parsed_dataset2 = raw_data2.map(_parse_function)
    parsed_dataset3 = raw_data3.map(_parse_function)
    parsed_dataset4 = raw_data4.map(_parse_function)
    parsed_dataset5 = raw_data5.map(_parse_function)
    parsed_dataset6 = raw_data6.map(_parse_function)
    parsed_dataset7 = raw_data7.map(_parse_function)
    parsed_dataset8 = raw_data8.map(_parse_function)
    parsed_val = raw_val.map(_parse_function)
    parsed_test = raw_test.map(_parse_function)

    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset1]
    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset2]
    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset3]
    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset4]
    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset5]
    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset6]
    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset7]
    [Xid.append(raw['kepid'].numpy()) for raw in parsed_dataset8]
    [Xvalid.append(raw['kepid'].numpy()) for raw in parsed_val]
    [Xtestid.append(raw['kepid'].numpy()) for raw in parsed_test]

    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset1]
    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset2]
    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset3]
    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset4]
    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset5]
    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset6]
    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset7]
    [Ytrain.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_dataset8]
    [Yval.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_val]
    [Ytest.append(one_hot(raw['av_training_set'].numpy())) for raw in parsed_test]

    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset1]
    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset2]
    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset3]
    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset4]
    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset5]
    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset6]
    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset7]
    [Xtrain.append(raw['global_view'].numpy()) for raw in parsed_dataset8]
    [Xval.append(raw['global_view'].numpy()) for raw in parsed_val]
    [Xtest.append(raw['global_view'].numpy()) for raw in parsed_test]

    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset1]
    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset2]
    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset3]
    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset4]
    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset5]
    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset6]
    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset7]
    [Xtrainl.append(raw['local_view'].numpy()) for raw in parsed_dataset8]
    [Xvall.append(raw['local_view'].numpy()) for raw in parsed_val]
    [Xtestl.append(raw['local_view'].numpy()) for raw in parsed_test]

    Xid=[('0000'+str(el)[:9])[-9:] for el in Xid]
    Xvalid=[('0000'+str(el)[:9])[-9:] for el in Xvalid]
    Xtestid=[('0000'+str(el)[:9])[-9:] for el in Xtestid]

    Xtrain=np.array(Xtrain)
    Xval=np.array(Xval)
    Xtest=np.array(Xtest)
    Xtrainl=np.array(Xtrainl)
    Xvall=np.array(Xvall)
    Xtestl=np.array(Xtestl)
    Ytrain=np.array(Ytrain)
    Ytest=np.array(Ytest)
    Yval=np.array(Yval)
    Xid=np.array(Xid)
    Xvalid = np.array(Xvalid)
    Xtestid = np.array(Xtestid)
    print(Xtrain.shape,Xval.shape,Xtest.shape)
    print(Xtrainl.shape,Xvall.shape,Xtestl.shape)
    print(Ytrain.shape,Yval.shape,Ytest.shape)
    print(Xid.shape, Xvalid.shape, Xtestid.shape)
    return(Xtrain, Xtrainl, Ytrain, Xid, Xval, Xvall, Yval, Xvalid, Xtest, Xtestl, Ytest, Xtestid)


#let me just stash a directory of residuals im gonna wanna check later on... got a good idea to shortlist planets and all
#just need to see if it works out in some way or not.
FILEPATH = 'E:\Masters_Project_Data\\alienworlds_fps\\'

def get_residuals_and_info():
    entries = os.listdir(FILEPATH)
    for ent in entries[4130:]:
        hdu = fits.open(FILEPATH+ent)
        av_entry=ascii.read(CATALOG+'autovetter_label.tab')
        av_pl=np.array(av_entry['tce_plnt_num'])
        ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
        ref_label=av_entry['av_training_set']
        seconds=av_entry['av_pred_class']

        loc=np.where(np.asarray(ref_kepid)==ent[4:13])
        if(len(loc[0]) == 0): continue

        m=len(hdu)
        try: flux2=hdu[m-1].data['RESIDUAL_LC']  
        except: continue 
        flux2 = np.asarray([0 if np.isnan(el) else el for el in flux2])
        flux2 = (flux2 - np.median(flux2)) / (np.median(flux2)-min(flux2))
        tps = [hdu[x].header['TPERIOD'] for x in range(1, m-1)]
        tdurs = [hdu[x].header['TDUR'] for x in range(1, m-1)]

        class_ar=[]
        for el in ref_label[loc[0]]:
            if(el=='PC'): class_ar.append(0)
            elif(el=='UNK'): class_ar.append(2)
            else: class_ar.append(1)

        pred_class_ar = [0 if(el=='PC') else 1 for el in seconds[loc[0]]]

        chunks = np.asarray([flux2[i:i+4000] for i in range(0,len(flux2)-4000,4000)])

        net = np.asarray([[chunks[i],tps,tdurs,m-2,class_ar, pred_class_ar] for i in range(len(chunks))], dtype='object')
        gc.write_tfr_record(TRAINING_MODULE+ent[4:13],net,
            ['input','tperiod','tdur','tce_num','av_training_set', 'av_pred_class'],
            ['ar','ar','ar','fl','ar','ar'],['float32','float16','float16','float16','int8','int8'])
        print("hit:", ent[4:13],chunks.shape, class_ar)



#get_labels_and_make_TS()
#take_out_from_dv()
#get_residuals_and_info()
#check_tfr_records()
cumulative_ts("../../processed_directories/sem_seg_ext/","../../training_data/cumulative_")
