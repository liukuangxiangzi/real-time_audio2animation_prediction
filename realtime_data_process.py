import os.path
import numpy as np
from keras.utils import to_categorical
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.special import softmax
import progressbar

def append_txt_data(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    txt_data = []
    for fname in sorted(os.listdir(dir)):
        if any(fname.endswith(extension) for extension in ['.txt']): #7790
            txt_file_data = np.array(open(os.path.join(dir, fname), "r").read().split())
            txt_data = np.append(txt_data, txt_file_data)
    print('txt_data_shape:', txt_data.shape)
    return txt_data
def append_np_data(dir, np_data_shape): #np_data_shape:(0,32,32) or (0,256)
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    np_data = np.array([], dtype=np.int64).reshape(np_data_shape) #animation_parameter input data:  Nframe x 256
    for fname in sorted(os.listdir(dir)):
        if any(fname.endswith(extension) for extension in ['.npy', '.NPY']): #(1420/1270, 32, 32)
            np_file_data = np.load(os.path.join(dir, fname))
            np_data = np.vstack([np_data, np_file_data])
    return np_data
#Viseme ID >>> Animation_parameter
def make_viseme_ID_dataset(dir):
    if os.path.exists(os.path.join(dir, 'Viseme_ID.npy')) is True:
        os.remove(os.path.join(dir, 'Viseme_ID.npy'))
    phone = append_txt_data(dir)
    phoneme_dict = {
        'ER0':9, 'ER1':9,  'AA1': 9, 'AA2':9, 'AE1': 9, 'AE2': 9, 'AH0': 9, 'AH1': 9, 'AH2': 9, 'AO1': 11, 'AW1': 9, 'AY1': 9,'AY2':9, 'B': 0, 'SH': 6, 'CH': 6, 'D': 5, 'DH': 3,
        'EH1': 9,'EH0':9, 'EH2':9, 'EY1': 9, 'IY2': 9,'F': 2, 'G': 7, 'HH': 8, 'IH0': 9, 'IH1': 9, 'IH2':9, 'IY0': 9,'IY1': 9, 'NG':9, 'JH': 6, 'K': 7, 'L': 4,
        'M': 0, 'N': 5, 'OW0': 11, 'OW1': 11, 'OW2': 11, 'P': 0, 'R': 5, 'S': 6, 'T': 5, 'TH': 3, 'UW1': 10, 'V': 2, 'W': 1, 'OY1':1,'UH1':10,
        'Y': 7, 'Z': 5, 'sil': 12, 'sp': 12
    }
    Viseme_ID = []
    for i in phone:
        v = phoneme_dict[i]
        Viseme_ID.append(v)
    Viseme_ID = to_categorical(Viseme_ID)
    print('Viseme_ID_shape', Viseme_ID.shape) #(7790,13)
    np.save(os.path.join(dir, 'viseme_ID'), Viseme_ID)
    print('viseme_ID_data saved in path %s' % dir)
#wav2vec feature >>> Animation_parameter
def make_w2v_dataset(dir, np_data_shape): #np_data_shape:(0,32,32)
    if os.path.exists(os.path.join(dir, 'w2v_data.npy')) is True:
        os.remove(os.path.join(dir, 'w2v_data.npy'))
    w2v_data = append_np_data(dir, np_data_shape)
    print("w2v_data_shape: %(s)s" % {'s': w2v_data.shape})
    np.save(os.path.join(dir, 'w2v_data.npy'), w2v_data)
    print('w2v_data saved in path %s' % dir)

def softmax_data(dir, file_name):
    # print(softmax(x_9[0][0]))
    data_softmax = np.load(os.path.join(dir, file_name))
    np.save(dir + '/softmax_' + file_name, data_softmax )
    print('softmax_%(n)s saved in path %(s)s.' % {'n': file_name[:-4], 's': dir})

def make_animation_parameter_dataset(dir, np_data_shape): #np_data_shape:(0,256)
    animation_parameter_dir = os.path.basename(dir)
    if os.path.exists(dir + '/' + animation_parameter_dir +'_animation_parameter.npy') is True:
        os.remove(os.path.join(dir, animation_parameter_dir +'_animation_parameter.npy'))
    animation_parameter_data = append_np_data(dir, np_data_shape)
    print("%(n)s_animation_parameter_data_shape: %(s)s" % {'n': animation_parameter_dir, 's': animation_parameter_data.shape})
    np.save(dir + '/' + animation_parameter_dir +'_animation_parameter.npy', animation_parameter_data)
    print("%(n)s_animation_parameter_data saved in path %(s)s" % {'n': animation_parameter_dir, 's': dir})

def make_train_test_dataset(dir, file_name): #save index in dir
    assert os.path.isfile(os.path.join(dir, file_name)), '%s is not a valid file name' % file_name
    if os.path.exists(dir + '/train_' + file_name) is True:
        os.remove(dir + '/train_' + file_name)
    if os.path.exists(dir + '/test_' + file_name) is True:
        os.remove(dir + '/test_' + file_name)
    data = np.load(os.path.join(dir, file_name))
    random.seed(10)
    train_index = sorted(random.sample(range(data.shape[0]), int(data.shape[0]*0.8)))
    test_index = list(set([*range(data.shape[0])]) - set(train_index))
    train_data = data[train_index,:]
    test_data = data[test_index,:]
    print('train_%(n)s_data_shape: %(s)s' % {'n': file_name[:-4],'s': train_data.shape})
    print('test_%(n)s_data_shape: %(s)s' % {'n': file_name[:-4],'s': test_data.shape})
    np.save(dir + '/train_' + file_name, train_data)
    np.save(dir + '/test_' + file_name, test_data)
    print('train and test %s generated.' % file_name[:-4])

def data_Normalization(dir, file_name):
    assert os.path.isfile(os.path.join(dir, file_name)), '%s is not a valid file name' % file_name
    data = np.load(os.path.join(dir, file_name))
    scaler = MinMaxScaler()
    scaled_data = np.array(scaler.fit_transform(data))
    np.save(dir + '/scaled_' + file_name, scaled_data)
    print('scaled_%(n)s saved in path %(s)s.' % {'n': file_name[:-4], 's': dir})












dir = '/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/test_data/animation-params/eyes'
file_name = 'eyes_animation_parameter.npy' #'w2v_data.npy'   #'viseme_ID.npy'
data_Normalization(dir,file_name)





