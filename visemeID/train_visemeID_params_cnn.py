import os
import logging
import argparse
import numpy as np
from datetime import datetime
from keras.utils.vis_utils import plot_model
from keras.layers import Input, LSTM, LeakyReLU, MaxPooling1D
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.layers import Dropout
from keras.optimizers import adam_v2
from keras.layers import Dense, Flatten, Conv1D

def configure_logging():
    logging.basicConfig(level=logging.INFO, filename='myapp.log', filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_input_data_dir, test_input_data_dir, train_eyes_data_dir, test_eyes_data_dir,
              train_mouth_data_dir, test_mouth_data_dir):
    X_train = np.load(train_input_data_dir)
    x_train = X_train.reshape(X_train.shape[0], 8, X_train.shape[1])
    print("***", x_train.shape)
    x_train = x_train.reshape(2416, 8, 13)
    print("*****", x_train.shape)
    X_test = np.load(test_input_data_dir)
    x_test = X_test.reshape(X_train.shape[0], 8, 13)
    y_eye_train = np.load(train_eyes_data_dir)
    y_eye_test = np.load(test_eyes_data_dir)
    y_mouth_train = np.load(train_mouth_data_dir)
    y_mouth_test = np.load(test_mouth_data_dir)
    return x_train, x_test, y_eye_train, y_eye_test, y_mouth_train, y_mouth_test

def define_model(x_train, resume_training, load_model_dir):
    if not resume_training:
        n_features = x_train.shape[2]
        lr = 1e-4
        initializer = 'glorot_uniform'

        # define trained_model
        net_in = Input(shape=(8, n_features))
        l1 = Conv1D(128, 3, padding='same', kernel_initializer=initializer, name='phoneme_Conv1D_1', activation='relu')(net_in)
        l1mp = MaxPooling1D()(l1)
        l2 = Conv1D(64, 3, padding='same', kernel_initializer=initializer, name='phoneme_Conv1D_2', activation='relu')(l1mp)
        l2mp = MaxPooling1D()(l2)
        l3 = Conv1D(32, 3, padding='same', kernel_initializer=initializer, name='phoneme_Conv1D_3', activation='relu')(l2mp)
        l3mp = MaxPooling1D()(l3)

        l3fl = Flatten()(l3mp)

        out_eye = Dense(256, kernel_initializer=initializer, name='out_eye', activation='linear')(l3fl)

        out_mouth = Dense(256, kernel_initializer=initializer, name='out_mouth', activation='linear')(l3fl)
        model = Model(inputs=net_in, outputs=[out_mouth, out_eye])
        model.summary()
        opt = adam_v2.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss={'out_mouth': 'mse', 'out_eye': 'mse'}, loss_weights={'out_mouth': 1.0, 'out_eye': 1.0},
                      metrics=['accuracy'])
    else:
        model = load_model(load_model_dir)
        logging.info('Trained_model loaded')
    return model

#tensorboard --logdir /Users/liukuangxiangzi/PycharmProjects/PhonemeNet/logs/fit/ --host=127.0.0.1
def train_model(model, x_train, y_mouth_train, y_eye_train, x_test, y_mouth_test, y_eye_test, epochs, batch_size, log_dir, save_model_dir):
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    logging.info('Starting training')
    history = model.fit(x=x_train,
                        y=[y_mouth_train, y_eye_train],
                        validation_data=(x_test, [y_mouth_test, y_eye_test]),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[tbCallBack],
                        shuffle=True)
    # Plot model structure
    plot_model(model, to_file='model_visemeID_params_cnn_plot.png', show_shapes=True, show_layer_names=True)
    # Log loss to console and file
    logging.info('Training loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['m_out_loss'][-1], history.history['e_out_loss'][-1]))
    logging.info('Validation loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['val_m_out_loss'][-1], history.history['val_e_out_loss'][-1]))
    # Save model to file
    model.save(os.path.join(save_model_dir, 'visemeID_cnn_epoch{}_bs{}.h5'.format(epochs, batch_size)))
    logging.info('Model saved to {}'.format(save_model_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-input-data-dir', type=str, default='data/phones/train_viseme_ID.npy', help='path to training input data')
    parser.add_argument('--test-input-data-dir', type=str, default='data/phones/test_viseme_ID.npy', help='path to test input data')
    parser.add_argument('--train-eyes-data-dir', type=str, default='data/animation-params/eyes/train_scaled_eyes.npy', help='path to eye animation params training data')
    parser.add_argument('--test-eyes-data-dir', type=str, default='data/animation-params/eyes/test_scaled_eyes.npy', help='path to eye animation params test data')
    parser.add_argument('--train-mouth-data-dir', type=str, default='data/animation-params/mouth/train_scaled_mouth.npy', help='path to mouth animation params training data')
    parser.add_argument('--test-mouth-data-dir', type=str, default='data/animation-params/mouth/test_scaled_mouth.npy', help='path to mouth animation params test data')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
    parser.add_argument('--resume-training', type=bool, default=False, help='whether to resume training from a saved model')
    parser.add_argument('--save-model-dir', type=str, default='visemeID/model/', help='directory where trained model will be saved')
    parser.add_argument('--load-model-dir', type=str, default=None, help='path to trained model to resume training from')
    args = parser.parse_args()

    configure_logging()
    x_train, x_test, y_eye_train, y_eye_test, y_mouth_train, y_mouth_test = load_data(args.train_input_data_dir,
                                                                                  args.test_input_data_dir,
                                                                                  args.train_eyes_data_dir,
                                                                                  args.test_eyes_data_dir,
                                                                                  args.train_mouth_data_dir,
                                                                                  args.test_mouth_data_dir)
    model = define_model(x_train, args.resume_training, args.load_model_dir)
    log_dir = os.path.join('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
    save_model_dir = args.save_model_dir
    train_model(model, x_train, y_mouth_train, y_eye_train, x_test, y_mouth_test, y_eye_test, args.epochs, args.batch_size, log_dir, save_model_dir)

if __name__ == '__main__':
    main()


# #tensorboard
# #tensorboard --logdir /Users/liukuangxiangzi/PycharmProjects/PhonemeNet/logs/fit/ --host=127.0.0.1
# log_dir = os.path.join(
#     "logs",
#     "fit",
#     datetime.now().strftime("%Y%m%d-%H%M%S"),
# )
# tbCallBack = TensorBoard(log_dir= log_dir, histogram_freq=0, write_graph=True, write_images=True)
#

#load data
# x = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/viseme_31_32_1188.npy') #(5648, 13*8)
# y_mouth = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/mouth_31_32_1188.npy') #(5648, 256)
# y_eyes = np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/eyes_31_32_1188.npy') #(5655, 256)

#x = X.reshape((-1, 8, 13))

# #load index
# train_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/train_index_1188.npy'))
# test_index = sorted(np.load('/Users/liukuangxiangzi/PycharmProjects/PhonemeNet/data_p2a/test_index_1188.npy'))
# x_train = x[train_index, :,:] #(4524, 13)
# x_test = x[test_index, :,:] #(1131, 13)
#
# y_m_train = y_mouth[train_index, :] #(4524, 256)
# y_m_test = y_mouth[test_index, :] #(1131, 256)
#
# y_e_train = y_eyes[train_index, :] #(4524, 256)
# y_e_test = y_eyes[test_index, :] #(1131, 256)


# #network
# epochs, batch_size, resume_training = 400, 16, False #32

# if not resume_training:
#     n_features = x_train.shape[2]
#     lr = 1e-4
#     initializer = 'glorot_uniform'
#
#     # define trained_model
#     net_in = Input(shape=(8, n_features))



#     l1 = Conv1D(128, 3,
#                 padding='same',
#                 kernel_initializer=initializer,
#                 name='phoneme_Conv1D_1',activation='relu')(net_in)
#     l1mp= MaxPooling1D()(l1)
#
#
#     l2 = Conv1D(64, 3,
#                 padding='same',
#                 kernel_initializer=initializer,
#                 name='phoneme_Conv1D_2',activation='relu')(l1mp)
#
#     l2mp = MaxPooling1D()(l2)
#     l3 = Conv1D(32, 3,
#                 padding='same',
#                 kernel_initializer=initializer,
#                 name='phoneme_Conv1D_3',activation='relu')(l2mp)
#     l3mp=MaxPooling1D()(l3)
#
#
#     l3fl = Flatten()(l3mp)
#
#     out_eye = Dense(256,
#                 kernel_initializer=initializer, name='out_eye',activation='linear')(l3fl)
#
#     out_mouth = Dense(256,
#                       kernel_initializer=initializer, name='out_mouth',activation='linear')(l3fl)
#
#     model = Model(inputs=net_in, outputs=[out_mouth,out_eye])
#     model.summary()
#     opt = Adam(lr=lr)
#     model.compile(optimizer=opt, loss={'out_mouth':'mse', 'out_eye':'mse'}, loss_weights={'out_mouth':1.0, 'out_eye':1.0},metrics=['accuracy'])
#     #plot_model(model, to_file= save_model_name + '.png', show_shapes=True, show_layer_names=True)
#
# else:
#     #model = load_model(save_model_dir + load_model_name +'.h5') #trained_model for 3layers cnn with x_z: audio2pho_model_ep30_1e-4_32_33sub_16khz_timestep512_cnn2
#     print('Trained_model loaded.')


model.fit(x = x_train,  #training_data_generator
          y = [y_m_train, y_e_train],
          validation_data= (x_test,[y_m_test,y_e_test]),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=[tbCallBack],
          shuffle=True)

plot_model(model, to_file='model_p2a_cnn_plot.png', show_shapes=True, show_layer_names=True)
#
#
model.save('p2a_cnn_ep400_31_32_1188' +'.h5')
print("Saved trained_model to disk.")