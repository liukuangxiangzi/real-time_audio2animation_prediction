import os
import numpy as np
from datetime import datetime
from keras.layers import Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import *
from keras.utils import plot_model
from keras.layers import Dense, Flatten, Conv1D
from keras.layers import BatchNormalization
from keras.activations import sigmoid
from keras.activations import relu
from sklearn import decomposition


def pca_decomposition(input):
    pca = decomposition.PCA(n_components=32, whiten=True)
    pca.fit(input)
    input = pca.transform(input)
    # pca_basis_scale = np.sqrt(pca.explained_variance_)
    # input = np.divide(input, pca_basis_scale)
    return input



#tensorboard
#tensorboard --logdir /Users/liukuangxiangzi/PycharmProjects/PhonemeNet/logs/fit/ --host=127.0.0.1
log_dir = os.path.join(
    "logs",
    "fit",
    datetime.now().strftime("%Y%m%d-%H%M%S"),
)
tbCallBack = TensorBoard(log_dir= log_dir, histogram_freq=0, write_graph=True, write_images=True)


#load data
x = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/w2v_1195_31_32_softmax.npy') #(5648, 13*8)
y_mouth = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/mouth_norm2_31_32_1195.npy')
y_eyes = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/eye_norm2_31_32_1195.npy')



#load index
train_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/train_index_1195.npy'))
test_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/test_index_1195.npy'))
x_train = x[train_index, :,:] #(4524, 13)
x_test = x[test_index, :,:] #(1131, 13)

y_m_train = y_mouth[train_index, :] #(4524, 256)
y_m_test = y_mouth[test_index, :] #(1131, 256)

y_e_train = y_eyes[train_index, :] #(4524, 256)
y_e_test = y_eyes[test_index, :] #(1131, 256)


# #network
epochs, batch_size, resume_training = 500, 32, False #32

# if not resume_training:
n_features = x_train.shape[2]
lr = 1e-4
initializer = 'glorot_uniform'

# define trained_model
net_in = Input(shape=(32, n_features))







l1 = Conv1D(128, 3,
            padding='same',
            kernel_initializer=initializer,
            name='phoneme_Conv1D_1')(net_in)
#l1bn = BatchNormalization(epsilon=1e-05, momentum=0.1, name='BatchNorm1')(l1)
l1af = sigmoid(l1)
l1mp= MaxPooling1D()(l1af)

l2 = Conv1D(64, 3,
            padding='same',
            kernel_initializer=initializer,
            name='phoneme_Conv1D_2')(l1mp)
l2af = sigmoid(l2)
#l2bn = BatchNormalization(epsilon=1e-05, momentum=0.1, name='BatchNorm2')(l2af)
l2mp = MaxPooling1D()(l2af)

l3 = Conv1D(32, 3,
            padding='same',
            kernel_initializer=initializer,
            name='phoneme_Conv1D_3')(l2mp)
#l3bn = BatchNormalization(epsilon=1e-05, momentum=0.1, name='BatchNorm3')(l3)
l3af = sigmoid(l3)
l3mp=MaxPooling1D()(l3af)

l3fl = Flatten()(l3mp)

out_eye = Dense(256,
                kernel_initializer=initializer, name='out_eye', activation='linear')(l3fl)
out_mouth = Dense(256,
                  kernel_initializer=initializer, name='out_mouth', activation='linear')(l3fl)


model = Model(inputs=net_in, outputs=[out_mouth, out_eye])
model.summary()
opt = Adam(lr=lr)
model.compile(optimizer=opt, loss={'out_mouth':'mse', 'out_eye':'mse'}, loss_weights={'out_mouth':1.0, 'out_eye':1.0},metrics=['accuracy'])

batch_per_epoch = int(x_train.shape[0]/batch_size)
for epoch in range(0, epochs):
    for j in range(batch_per_epoch):
        batch_x = x_train[j*batch_size:(j+1)*batch_size]
        batch_y_e = y_e_train[j*batch_size:(j+1)*batch_size]
        batch_y_m = y_m_train[j*batch_size:(j+1)*batch_size]
        if j == 0:
            loss, loss_m,loss_e,acc_m,acc_e = model.train_on_batch(batch_x, [batch_y_m,batch_y_e])


        else:
            last_y_m, last_y_e =  model.predict(x_train[(j-1)*batch_size:j*batch_size])
            last_y_concat = np.concatenate((last_y_m,last_y_e),axis=-1)
            last_y_pca = pca_decomposition(last_y_concat)

            batch_x[:,0,:] = last_y_pca[:,:]
            loss, loss_m,loss_e,acc_m,acc_e = model.train_on_batch(batch_x, [batch_y_m,batch_y_e])

        print('>%d, %d/%d, loss=%.3f, loss_m=%.3f, loss_e=%.3f, acc_m=%.3f, acc_e=%.3f' % (epoch+1, j+1, batch_per_epoch, loss, loss_m,loss_e,acc_m,acc_e))


    #
    # val_loss = []
    # for batch, (X, y) in enumerate(test_data):
    #     val_loss.append(validate_on_batch(X, y))
    #
    # print('Validation Loss: ' + str(np.mean(val_loss)))



plot_model(model, to_file='model_p2a_cnn_replace_plot.png', show_shapes=True, show_layer_names=True)
#
#
model.save('previous_animation_sofmax_3siglinear' +'.h5')
print("Saved trained_model to disk.")

