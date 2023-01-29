import os
import numpy as np
from datetime import datetime
from keras.utils.vis_utils import plot_model
from data_process import utils
from keras.layers import Input, LSTM, LeakyReLU
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
x = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/viseme_5648.npy') #(5655, 13)
y_mouth = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/mouth_5648.npy') #(5655, 256)
y_eyes = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/eyes_5648.npy') #(5655, 256)



#load index
train_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/train_index_5648.npy'))
test_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/test_index_5648.npy'))
x_train = x[train_index, :] #(4524, 13)
x_test = x[test_index, :] #(1131, 13)

y_m_train = y_mouth[train_index, :] #(4524, 256)
y_m_test = y_mouth[test_index, :] #(1131, 256)

y_e_train = y_eyes[train_index, :] #(4524, 256)
y_e_test = y_eyes[test_index, :] #(1131, 256)


# #network
epochs, batch_size, resume_training = 300, 32, False #32

if not resume_training:
    n_features = x_train.shape[1]
    lr = 1e-4
    initializer = 'glorot_uniform'

    # define trained_model
    net_in = Input(shape=(n_features))



    l1 = Dense(128,
               kernel_initializer=initializer, name='l1')(net_in)
    lrelu1 = LeakyReLU(0.2)(l1)

    l2 = Dense(256,
               kernel_initializer=initializer, name='l2')(net_in)

    lrelu2 = LeakyReLU(0.2)(l2)

    l3 = Dense(512,
               kernel_initializer=initializer, name='l3')(lrelu2)

    lrelu2_2 = LeakyReLU(0.2)(l3)

    #flatten = Flatten()(lrelu2_2)
    ###############
    d1 = Dense(512,
               kernel_initializer=initializer, name='d1')(lrelu2_2)
    lrelu3 = LeakyReLU(0.2)(d1)

    d2 = Dense(256,
               kernel_initializer=initializer, name='d2')(lrelu3)
    lrelu4 = LeakyReLU(0.2)(d2)

    d3 = Dense(256,
               kernel_initializer=initializer, name='d_out',activation='linear')(lrelu4)

    e1 = Dense(512,
               kernel_initializer=initializer, name='e1')(lrelu2_2)
    elrelu3 = LeakyReLU(0.2)(e1)

    e2 = Dense(256,
               kernel_initializer=initializer, name='e2')(elrelu3)
    elrelu4 = LeakyReLU(0.2)(e2)

    e3 = Dense(256,
               kernel_initializer=initializer, name='e_out',activation='linear')(elrelu4)#

    model = Model(inputs=net_in, outputs=[d3,e3])
    model.summary()
    opt = Adam(lr=lr)
    model.compile(optimizer=opt, loss={'d_out':'mse', 'e_out':'mse'}, loss_weights={'d_out':1.0, 'e_out':1.0},metrics=['accuracy'])
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

plot_model(model, to_file='model_p2a_plot.png', show_shapes=True, show_layer_names=True)
#
#
model.save('p2a_mlp_ep300_5648' +'.h5')
print("Saved trained_model to disk.")