import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import os
from scipy.interpolate import interp1d
from astropy.io import fits,ascii
import sys

sys.path.append('C:\\Users\\Hp\\Documents\FYProj\\alienworlds\\Data_Processing')
import GetLightcurves as gc

"""TESS NAVIGATION MODULE
This program will enable one to sift through TESS data corresponding to single sector DV 
Timeseries. The module aims to return various datasets derived from the directory such as
folded or unfolded ligtcurves, binned or unbinned lightcurves etc.

PS: Will edit out the Kepler Code as well, later.


"""

filepath = '/media/ushasi/Elements/Masters_Project_Data/TESS/'
filepath = 'E:\\Masters_Project_Data\\TESS\\'
CATALOG = "../../Catalogs/"

def remove_nan(red_flux):
    for i in range(0,len(red_flux)):
        if np.isnan(red_flux[i]):
            red_flux[i]=0

def sort_data(hdu, hduno=1, crop=1.0, phase_length=2, binsize='None', centred=True):
    """ gives us a phase-folded, sorted, centered LC from a fits file 

    :param hdu: obtained from fits.open(file)
    :param hduno: the TCE number in the file
    :param crop: trim the edges of the LC. crop=1 means any phase value greater than 1 or less    than -1 is discarded. If set to 'lv' returns the lightcurve in local view. If set as a tuple, eg (-1,0) anything with phase less than -1 and greater than 0 will be cropped
    :param phase_length: length of the phase interval. 2 means phase of -1,1, 1 means phase interval of -0.5,0.5
    :binsize: If set to 'None' no binning takes place. Else, it is binned to the number of pixels
    specified.
    :param centred: This parameter centres the transit to the middle of TCE. If set as false it takes the default position given in HDU.

    returns - 
    a pandas dataframe containing details of stuff,
    transit period(in days),
    transit duration(in days),
    noise, in the form of standard deviation

    """
    
    flux = []
    phase = []
    try: 
        tp = hdu[hduno].header['TPERIOD']
        td = hdu[hduno].header['TDUR']/24
    except: return(0,0,0,0)
    for ph, fl in zip(hdu[hduno].data['PHASE'],hdu[hduno].data['LC_DETREND']):
        if not np.isnan(fl):
            flux.append(fl)
            if(ph/tp>0.5 and centred): phase.append(ph*phase_length/tp -phase_length)
            else: phase.append(ph*phase_length/tp)
    dfunb = pd.DataFrame(list(zip(phase, flux)),columns=['phase', 'flux'])
    df=dfunb.sort_values('phase',axis=0,ascending=True)

    if(crop=='lv'):
        df=df[(df.phase>-phase_length*td/tp) & (df.phase<phase_length*td/tp)] 
        
    elif(type(crop)==tuple): df=df[(df.phase>crop[0]) & (df.phase<crop[1])]

    else: df=df[(df.phase>-crop) & (df.phase<crop)]

    if(binsize!='None'):
        bins=np.linspace(min(df['phase']),max(df['phase']),binsize)
        groups = df.groupby(np.digitize(df['phase'], bins))
        df=groups.median()
        
        if(len(df)<binsize):
            func = interp1d(df['phase'], df['flux'])
            ph = np.linspace(min(df['phase']), max(df['phase']), binsize)
            fl = func(ph)
            df = pd.DataFrame(zip(ph, fl), columns=['phase','flux'])
            

    df_noise = df[(df.phase<-3*td/tp) | (df.phase>3*td/tp)]
    noise = np.std(np.array(df_noise['flux']))
    return(df, hdu[hduno].header['TPERIOD'], hdu[hduno].header['TDUR']/24, noise)

def segmentation_map(hdu, lab_arr, bins, clean_param = 1.0, skip_all_bkg=True):
    """ Returns a segmentation map for a given DV Timeseries
    
    """

    try: flux = hdu[1].data['LC_DETREND']
    except: return(0, "can't open data... corrupt hdu")
    try: residue = hdu[len(hdu)-1].data['RESIDUAL_LC']
    except: return(0, "can't open data... corrupt hdu")

    #get a preliminary phase... must center at least one transit
    ind_arr=np.arange(0,len(flux)-bins,bins)

    lightcurve=[]
    totmask=[]
    counts=[]
    tdurs=[hdu[i].header['TDUR'] for i in range(1,len(hdu)-2)]
    tps=[hdu[i].header['TPERIOD'] for i in range(1,len(hdu)-2)]
    tds=[hdu[i].header['TDEPTH'] for i in range(1,len(hdu)-2)]
    
    for ind in ind_arr:
        counting=[0,0]
        red_flux=flux[ind:ind+bins]
        red_res=residue[ind:ind+bins]

        if(len(red_flux)==0): continue

        #get a clean chunk
        count_nan=np.isnan(red_flux).sum() 
        if(count_nan/bins > clean_param): 
           continue

        #whatever is in the first raw LC and is 'nan' in the last residue plot is a map. mark those as [1,1,1] and the rest as background [0,0,0]
        mask=np.asarray([[1,1,1] if (np.isnan(red_res[x]) and not np.isnan(red_flux[x])) else [0,0,1] for x in range(0,len(red_res))])

        #eliminate the chunk if it contains only backgrounds.
        if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all() and skip_all_bkg): 
            continue
        
        #trackrog has pixels that were 'nan' to begin with meaning unobserved data and all.
        trackrog=np.where(np.isnan(red_flux))[0]
        for b in trackrog:
            mask[b]=[0,0,1]
        remove_nan(red_flux)
            
        for tce in range(1,len(hdu)-2):

            #checking the phase array to confirm if there are any detected transits in the chunk
            red_phase=hdu[tce].data['PHASE'][ind:ind+bins]
            ph_ind=[i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)]
            if(len(ph_ind)==0): continue

            #setting a label for a certain tce header iteration
            try:
                if(lab_arr[tce-1]=='PC'): 
                        label=[1,0,0]
                        counting[0]+=1
                elif(lab_arr[tce-1]=='AFP' or lab_arr[tce-1]=='NTP'): 
                        label=[0,1,0]
                        counting[1]+=1
                else: 
                    label=[0,1,0]
                    counting[1]+=1
            except ValueError as ve:
                #print("miss ind:",el[4:13])
                label=[0,0,1]
                
            new_flux=hdu[tce+1].data['LC_DETREND']
            new_flux=new_flux[ind:ind+bins]

            #this is the main loop setting the mask values to a certain label in an iteration
            for m in range(0,len(mask)):
                if(np.isnan(new_flux[m]) and (np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                    mask[m]=label

        #the last iteration that takes from the residue data part
        red_phase=hdu[len(hdu)-2].data['PHASE'][ind:ind+bins]
        ph_ind=[i for i in range(1,bins) if (red_phase[i]*red_phase[i-1]<0 and np.abs(red_phase[i]*red_phase[i-1])<0.001)]
        if(len(ph_ind)>0): 
            try:
                if(lab_arr[tce-1]=='PC'): 
                        label=[1,0,0]
                        counting[0]+=1
                elif(lab_arr[tce-1]=='AFP' or lab_arr[tce-1]=='NTP'): 
                        label=[0,1,0]
                        counting[1]+=1
                else: 
                    label=[0,1,0]
                    counting[1]+=1
            except ValueError as ve:
                #print("miss ind:",el[4:13])
                label=[0,0,1]

            for m in range(0,len(mask)):
                    if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                        mask[m]=label
        else: 
            #if the last iteration marks nothing useful, then replace everything as background
            for m in range(0,len(mask)):
                if((np.asarray(mask[m])==np.asarray([1,1,1])).all()):
                    mask[m]=[0,0,1]

        #checking once again if the chunk has only backgrounds marked in it
        if(np.asarray([(np.asarray(m)==np.asarray([0,0,1])).all() for m in mask]).all() and skip_all_bkg): continue
        lightcurve.append(red_flux)
        totmask.append(mask.reshape(-1))
        counts.append(counting)
            

    if(len(lightcurve)==0):
        return(0, 'no segmentation maps detected')   

    op_dict = {
        'lc':lightcurve,
        'mask':mask,
        'counts':counts,
        'tdurs':tdurs,
        'tps':tps,
        'tds':tds
    }
    
    return(op_dict, "success")


def segmented_directory(pathin, pathout, bins):
    """ This function creates a directory of objects where each file contains a lightcurve 
    along with a segmentation map. Useful to create training samples for the semantic segmentation
    experiment.

    
    """
    entries=os.listdir(pathin)
    av_entry=ascii.read(CATALOG+'autovetter_label.tab')
    av_pl=np.array(av_entry['tce_plnt_num'])
    ref_kepid=[('0000'+str(el)[:9])[-9:] for el in av_entry['kepid']]
    ref_label=av_entry['av_training_set']
    seconds=av_entry['av_pred_class']

    tick = 0
    for el in entries:
        tick+=1 
        #if(tick==20): break 
        hdu = fits.open(pathin+el)
        
        #creating the array of labels
        labarr=[]
        loc=np.where(np.asarray(ref_kepid)==el[4:13])
        for n in range(1, len(hdu)-1):
            loc_f=[m for m in loc[0] if str(av_pl[m])==str(n)]
            if(len(loc_f)==0):
                labarr.append('BKG')
            elif(ref_label[loc_f[0]]=='UNK'):
                labarr.append(seconds[loc_f[0]])
            else:
                labarr.append(ref_label[loc_f[0]])
                

        segmap, err = segmentation_map(hdu, labarr, bins, 0.1)

        print(el[4:13], err)

        if(err=='success'):
            lc = segmap['lc']
            mask = segmap['mask']
            cnt = segmap['counts']
            tdur = segmap['tdurs']
            tps = segmap['tps']
            tds = segmap['tds']
            net = np.asarray([[lc[i],mask[i],cnt[i],tdur,tps,tds] for i in range(0,len(counts))], dtype='object')
        gc.write_tfr_record(pathout+el[4:13],net,
            ['input','mask','counts','tdur','tperiod','tdepth'],['ar','ar','ar','ar','ar','ar'],
            ['float32','bool', 'int8','float16','float16','float16'])
        print(tick,'hit:',el[4:13])



#--------------------------------------------------------------------------------
# bulk extractions here for local and global views. Follow this up with segmentation
# routines, and routines that convert this into test sets for the bezier shape excercise.

def TESS_single_sector(opfile, sector, bins=2000, crop=0.5, phaselength=1):
    """ returns phase-folded, potentially centred, scaled, cropped and binned version of
    all lightcurves in a TESS sector in the form of a TFRrecords file useful as a test sample
    for planet detection or arbitrary shape detection.

    :param opfile: Name of output file along with the required path
    :param sector: sector number to convert to TFRrecord
    :param bins: In case of binning, the size of the bin. Default - 2000. 'None' is also a valid 
    attribute. In this case the original length of LC is retained.
    :param crop: Trim the LC.
    :param phaselength: The total length of phase that spans the orbital duration, e.g. 1,2,2pi etc.

    Returns:-
    None, but saves the associated TFRrecords file in the chosen directory.

    Note:-
    If using the 'read_tfr_record' function from the GetLightcurves module, use the following settings:
    feature_map = ['flux', 'phase', 'id', 'pl_no', 'tperiod', 'tdur']
    data_type = ['ar', 'ar', 'b', 'i', 'fl', 'fl']
    fin_type = [tf.float32, tf.float32, tf.string, tf.int8, tf.float32, tf.float32]

    """

    entries = os.listdir(filepath+'sector'+str(sector)+'/')
    netfl =[]
    netph =[]
    netid =[]
    netno =[]
    nettp =[]
    nettd =[]

    for entry in entries:
        hdu = fits.open(filepath+'sector'+str(sector)+'/'+entry)
        tid = entry[30:46]
        n = len(hdu)
        for i in range(1, n-1):
            print('kid: ', tid, i)
            try: df, tp, td, _ = sort_data(hdu, i, crop, phaselength, bins)
            except: 
                print('miss', tid)
                continue
            netfl.append(np.array(df['flux']))
            netph.append(np.array(df['phase']))
            netid.append(tid)
            netno.append(i)
            nettp.append(tp)
            nettd.append(td)

    net = [[netfl[i], netph[i], netid[i], netno[i], nettp[i], nettd[i]] for i in range(0,len(netid))]
    cols = ['flux', 'phase', 'id', 'pl_no', 'tperiod', 'tdur']
    datatype = ['ar', 'ar', 'b', 'i', 'fl', 'fl']
    fintype = ['float32', 'float32', 'string', 'int8', 'float32', 'float32']

    gc.write_tfr_record(opfile, net, cols, datatype, fintype)

def TESS_single_sector_local_global(opfile, sector, lv_bins=200, gv_bins=2000):
    """ emulates the local and global views defined by Shallue and Vanderberg Neural Network.

    :param opfile: Name of output file along with the required path
    :param sector: sector number to convert to TFRrecord
    :param lv_bins: Number of bins in local view. Default - 200. 
    :param gv_bins: NUmber of bins in global view. Default - 2000.

    Returns:-
    None, but saves the associated TFRrecords file in the chosen directory.
    
    """

    entries = os.listdir(filepath+'sector'+str(sector)+'/')
    netfl_l =[]
    netfl_g =[]
    netid =[]
    netno =[]
    nettp =[]
    nettd =[]

    for entry in entries:
        hdu = fits.open(filepath+'sector'+str(sector)+'/'+entry)
        tid = entry[30:46]
        n = len(hdu)
        for i in range(1, n-1):
            print('kid: ', tid, i)
            try: df, tp, td, _ = sort_data(hdu, i, 'lv', 1, lv_bins)
            except:
                print('miss', tid)
                continue
            try: dfg, tp, td, _ = sort_data(hdu, i, 0.5, 1, gv_bins, False)
            except:
                print('miss', tid)
                continue
            netfl_l.append(np.array(df['flux']))
            netfl_g.append(np.array(dfg['flux']))
            netid.append(tid)
            netno.append(i)
            nettp.append(tp)
            nettd.append(td)

    net = [[netfl_l[i], netfl_g[i], netid[i], netno[i], nettp[i], nettd[i]] for i in range(0,len(netid))]
    cols = ['flux_loc', 'flux_glob', 'id', 'pl_no', 'tperiod', 'tdur']
    datatype = ['ar', 'ar', 'b', 'i', 'fl', 'fl']
    fintype = ['float32', 'float32', 'string', 'int8', 'float32', 'float32']

    gc.write_tfr_record(opfile, net, cols, datatype, fintype)

def TESS_segmented_raw(opfile, sector, bins=4000):
    entries = os.listdir(filepath+'sector'+str(sector)+'/')
    netfl =[]
    netid =[]

    for entry in entries:
        hdu = fits.open(filepath+'sector'+str(sector)+'/'+entry)
        tid = entry[30:46]

        print('Processing kid ...', tid)

        fl = hdu[1].data['LC_DETREND']
        chunks = [fl[i:i+bins] for i in range(0, len(fl)-bins, bins)]
        netfl.append(chunks)
        netid.append(tid)

    net = [[netfl[i], netid[i]] for i in range(0,len(netid))]
    cols = ['flux', 'id']
    datatype = ['ar', 'b', 'i']
    fintype = ['float32', 'string']

    gc.write_tfr_record(opfile, net, cols, datatype, fintype)


#--------------------------------------------------------------------------------   

TESS_single_sector('../../training_data/TESS_sector1_bezier', 1, 500, 'lv')
