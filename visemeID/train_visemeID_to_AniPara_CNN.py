import os
import numpy as np
from datetime import datetime
from keras.utils.vis_utils import plot_model
from data_process import utils
from keras.layers import Input, LSTM, LeakyReLU, MaxPooling1D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.layers import Dropout
from keras.optimizers import *
from keras.utils import plot_model
from keras.layers import Dense, Flatten, Conv1D




#tensorboard
#tensorboard --logdir /Users/liukuangxiangzi/PycharmProjects/PhonemeNet/logs/fit/ --host=127.0.0.1
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.now().strftime("%Y%m%d-%H%M%S"),
)
tbCallBack = TensorBoard(log_dir= log_dir, histogram_freq=0, write_graph=True, write_images=True)


#load data
x = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/viseme_31_32_1188.npy') #(5648, 13*8)
y_mouth = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/mouth_31_32_1188.npy') #(5648, 256)
y_eyes = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/eyes_31_32_1188.npy') #(5655, 256)

x = x.reshape(1188, 8, 13)

#load index
train_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/train_index_1188.npy'))
test_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/test_index_1188.npy'))
x_train = x[train_index, :,:] #(4524, 13)
x_test = x[test_index, :,:] #(1131, 13)

y_m_train = y_mouth[train_index, :] #(4524, 256)
y_m_test = y_mouth[test_index, :] #(1131, 256)

y_e_train = y_eyes[train_index, :] #(4524, 256)
y_e_test = y_eyes[test_index, :] #(1131, 256)


# #network
epochs, batch_size, resume_training = 400, 16, False #32

if not resume_training:
    n_features = x_train.shape[2]
    lr = 1e-4
    initializer = 'glorot_uniform'

    # define trained_model
    net_in = Input(shape=(8, n_features))



    l1 = Conv1D(128, 3,
                padding='same',
                kernel_initializer=initializer,
                name='phoneme_Conv1D_1',activation='relu')(net_in)
    l1mp= MaxPooling1D()(l1)


    l2 = Conv1D(64, 3,
                padding='same',
                kernel_initializer=initializer,
                name='phoneme_Conv1D_2',activation='relu')(l1mp)

    l2mp = MaxPooling1D()(l2)
    l3 = Conv1D(32, 3,
                padding='same',
                kernel_initializer=initializer,
                name='phoneme_Conv1D_3',activation='relu')(l2mp)
    l3mp=MaxPooling1D()(l3)


    l3fl = Flatten()(l3mp)

    out_eye = Dense(256,
                kernel_initializer=initializer, name='out_eye',activation='linear')(l3fl)

    out_mouth = Dense(256,
                      kernel_initializer=initializer, name='out_mouth',activation='linear')(l3fl)

    model = Model(inputs=net_in, outputs=[out_mouth,out_eye])
    model.summary()
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss={'out_mouth':'mse', 'out_eye':'mse'}, loss_weights={'out_mouth':1.0, 'out_eye':1.0},metrics=['accuracy'])
    #plot_model(model, to_file= save_model_name + '.png', show_shapes=True, show_layer_names=True)

else:
    #model = load_model(save_model_dir + load_model_name +'.h5') #trained_model for 3layers cnn with x_z: audio2pho_model_ep30_1e-4_32_33sub_16khz_timestep512_cnn2
    print('Trained_model loaded.')


model.fit(x = x_train,  #training_data_generator
          y = [y_m_train, y_e_train],
          validation_data= (x_test,[y_m_test,y_e_test]),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[tbCallBack],
          shuffle=True  #False

          )

plot_model(model, to_file='model_p2a_cnn_plot.png', show_shapes=True, show_layer_names=True)
#
#
model.save('p2a_cnn_ep400_31_32_1188' +'.h5')
print("Saved trained_model to disk.")