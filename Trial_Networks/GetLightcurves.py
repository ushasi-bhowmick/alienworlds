#trying to cleanup my code and build modules that are robust enough to be useful...
#these functions can be used to write tfr record files.

#The user - interface functions are: read_tfr_record and write_tfr_record

import numpy as np 
import tensorflow as tf

#these are some feature definitions
def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
  array = tf.io.serialize_tensor(array)
  return array

def gen_write_feature(content,feature_map,data_type, fin_type):
    fn_arr=[]
    for i in range(0,len(feature_map)):
        if(data_type[i]=='ar'):
            fn_arr.append(_bytes_feature(serialize_array(np.asarray(content[i], dtype=fin_type[i]))))
        elif(data_type[i]=='fl'):
            fn_arr.append(_float_feature(content[i]))
        elif(data_type[i]=='i'):
            fn_arr.append(_int64_feature(content[i]))
        elif(data_type[i]=='b'):
            fn_arr.append(_bytes_feature(serialize_array(content[i])))

    desc = dict(zip(feature_map, fn_arr))
    out = tf.train.Example(features=tf.train.Features(feature=desc))
    return(out)

# This is used to write a tfr record file. Add:
# a filename,
# the entire content as an n dimensional array,
# the feature map array which is the key for each column
# data_type, which defines the type while serialising data for storage. one of 4 options array, float, int, byte (keywords: ar, fl, i, b)
# fin_type, the actual data type of data, which determines the memory of stored data, a valid np data type
def write_tfr_record(filename,content,feature_map,data_type, fin_type):
    if(len(content[0]) != len(feature_map) != len(data_type)):
        print('inconsistent sampling')
        return 0

    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store our data to disk
    count = 0

    #loop over content
    for el in content:
        out = gen_write_feature(el, feature_map, data_type, fin_type)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")
    return count

def gen_read_feature(content, feature_map, data_type,fin_type):
    fn_arr=[]
    for i in range(0,len(feature_map)):
        if(data_type[i]=='ar'):
            fn_arr.append(tf.io.FixedLenFeature([], tf.string))
        elif(data_type[i]=='fl'):
            fn_arr.append(tf.io.FixedLenFeature([], tf.float32))
        elif(data_type[i]=='i'):
            fn_arr.append(tf.io.FixedLenFeature([], tf.int64))
        elif(data_type[i]=='b'):
            fn_arr.append(tf.io.FixedLenFeature([], tf.string))

    desc = dict(zip(feature_map, fn_arr))
    ex_msg = tf.io.parse_single_example(content, desc)

    output = []
    for i in range(0,len(feature_map)):
        #output.append(tf.io.parse_tensor(ex_msg[feature_map[i]], out_type=fin_type[i]))
        if(data_type[i]=='ar' or data_type[i]=='b'):
           output.append(tf.io.parse_tensor(ex_msg[feature_map[i]], out_type=fin_type[i]))
        #elif(data_type[i]=='b'):
        #   output.append(tf.io.parse_tensor(ex_msg[feature_map[i]], out_type=tf.string))
        else:
           output.append(ex_msg[feature_map[i]])
    
    return(output)
 
# This is used to read a tfr record file. Add:
# a filename,
# the feature map array which is the key for each column (as defined previously in tfr record file)
# data_type, which defines the type while serialising data for storage. one of 4 options array, float, int, byte (keywords: ar, fl, i, b)
# fin_type, the actual data type of data, which determines the memory of stored data, a valid tensorflow data type
# the output is returned in form of numpy arrays one for each feature.
def read_tfr_record(filename, feature_map, data_type, fin_type):
    tfr_dataset = tf.data.TFRecordDataset([filename]) 
    dataset = tfr_dataset.map(lambda x: gen_read_feature(x, feature_map, data_type, fin_type))
    print(dataset)
    output = []
    for el in dataset:
       output.append(el)
    output2=[]
    
    for i in range(0,len(feature_map)):
        temp = []
        for el in output:
            temp.append(np.asarray(el[i]))
        output2.append(temp)
    
    
    return(output2)

# this function call will extract the data from the results directory
ip,tp,pp,sm,ss,plp,fpsp = read_tfr_record('../../processed_directories/expand_test_result/001026133',
    ['input','true_map','pred_map','scale_median','scale_std','pl_peaks','fps_peaks'],
    ['ar','ar','ar','fl','fl','ar','ar'], 
    [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int16, tf.int16])

#an example function to write a tfr record.
write_tfr_record('testthecode',[[32.5,[3,3,3]],[22.66,[2,2,2]],[333.4,[8,8,8]]],
    ['id1','id2'],['fl','ar'],['float32','int8'])


