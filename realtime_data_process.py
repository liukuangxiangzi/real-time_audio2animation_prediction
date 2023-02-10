import os.path
import numpy as np
from tensorflow.keras.utils import to_categorical
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from scipy.special import softmax
#import progressbar
import argparse

#append phones. dir: directory of phones data folder
def append_txt_data(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    txt_data = []
    for fname in sorted(os.listdir(dir)):
        if any(fname.endswith(extension) for extension in ['.txt']):
            txt_file_data = np.array(open(os.path.join(dir, fname), "r").read().split())
            txt_data = np.append(txt_data, txt_file_data)
    print("txt_data_shape: %(s)s" % {'s': txt_data.shape})
    return txt_data

#phones to Viseme ID. dir: directory of phones data folder
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
    print('Viseme_ID_shape', Viseme_ID.shape)
    np.save(os.path.join(dir, 'viseme_ID'), Viseme_ID)
    print('viseme_ID_data saved in path %s' % dir)



#####
#append animation params, wave2vec features. dir: directory of eyes/mouth animation params data or wave2vec data folder
def append_np_data(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    np_data = np.array([], dtype=np.int64)
    file_name = os.path.basename(dir)
    if os.path.exists(dir + '/' + file_name + '.npy') is True:
        os.remove(os.path.join(dir, file_name + '.npy'))
    for fname in sorted(os.listdir(dir)):
        if any(fname.endswith(extension) for extension in ['.npy', '.NPY']):
            np_file_data = np.load(os.path.join(dir, fname))
            if np_data.size == 0:
                np_data = np_file_data
            else:
                np_data = np.vstack([np_data, np_file_data])
    print("np_data_shape: %(s)s" % {'s': np_data.shape})
    np.save(dir + '/' + file_name + '.npy', np_data)
    print("%(n)s_data saved in path %(s)s" % {'n': file_name, 's': dir})
    return np_data


#calculate the softmax of wave2vec data.  dir: directory of wave2vec.npy file
def softmax_data(file_path):
    file_name = os.path.basename(file_path)
    data_softmax = softmax(np.load(file_path))
    np.save(file_path[:-4] + '_softmax.npy', data_softmax )
    print('softmax_%(n)s saved in path %(s)s.' % {'n': file_name[:-4], 's': os.path.dirname(file_path)})

#dir: directory of viseme_ID.npy, wave2vec.npy, eyes.npy or mouth.npy file
def make_train_test_dataset(file_path):
    assert os.path.isfile(file_path), '%s is not a valid file name' % file_path
    dir, file_name = os.path.split(file_path)
    if os.path.exists(dir + '/train_' + file_name) is True:
        os.remove(dir + '/train_' + file_name)
    if os.path.exists(dir + '/test_' + file_name) is True:
        os.remove(dir + '/test_' + file_name)
    data = np.load(file_path)
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


#dir: directory of eyes.npy or mouth.npy file
def data_Normalization(file_path):
    assert os.path.isfile(file_path), '%s is not a valid file name' % file_path
    dir, file_name = os.path.split(file_path)
    if os.path.exists(dir + '/scaled_' + file_name) is True:
        os.remove(dir + '/scaled_' + file_name)
    data = np.load(file_path)
    scaler = MinMaxScaler()
    scaled_data = np.array(scaler.fit_transform(data))
    print('scaled_%(n)s_data_shape: %(s)s' % {'n': file_name[:-4], 's': scaled_data.shape})
    np.save(dir + '/scaled_' + file_name, scaled_data)
    print('scaled_%(n)s saved in path %(s)s.' % {'n': file_name[:-4], 's': dir})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run functions for processing data.')
    parser.add_argument('function', type=str, choices=['append_txt_data', 'make_viseme_ID_dataset', 'append_np_data', 'softmax_data', 'make_train_test_dataset', 'data_Normalization'], help='Name of the function to run')
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory containing data')
    args = parser.parse_args()

    if args.function == 'append_txt_data':
        append_txt_data(args.directory)
    elif args.function == 'make_viseme_ID_dataset':
        make_viseme_ID_dataset(args.directory)
    elif args.function == 'append_np_data':
        append_np_data(args.directory)
    elif args.function == 'softmax_data':
        softmax_data(args.directory)
    elif args.function == 'make_train_test_dataset':
        make_train_test_dataset(args.directory)
    elif args.function == 'data_Normalization':
        data_Normalization(args.directory)














