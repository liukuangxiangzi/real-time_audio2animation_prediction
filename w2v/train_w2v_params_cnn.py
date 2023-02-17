import os
import logging
import argparse
import numpy as np
from datetime import datetime
from keras.layers import Input, MaxPooling1D
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Flatten, Conv1D
from keras.layers import BatchNormalization
from keras.activations import sigmoid
from keras.activations import relu
from keras.layers import Dropout, add
from sklearn import decomposition




def configure_logging():
    logging.basicConfig(level=logging.INFO, filename='myapp.log', filemode='w',
                        format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_input_data_dir, test_input_data_dir, train_eyes_data_dir, test_eyes_data_dir,
              train_mouth_data_dir, test_mouth_data_dir):
    x_train = np.load(train_input_data_dir)
    x_test = np.load(test_input_data_dir)
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
        net_in = Input(shape=(32, n_features))

        l1 = Conv1D(128, 3,
                    padding='same',
                    kernel_initializer=initializer,
                    name='phoneme_Conv1D_1')(net_in)
        l1af = relu(l1)
        # l1bn = BatchNormalization(epsilon=1e-05, momentum=0.1, name='BatchNorm1')(l1af)
        l1mp = MaxPooling1D()(l1af)

        l2 = Conv1D(64, 3,
                    padding='same',
                    kernel_initializer=initializer,
                    name='phoneme_Conv1D_2')(l1mp)
        l2af = sigmoid(l2)
        # l2bn = BatchNormalization(epsilon=1e-05, momentum=0.1, name='BatchNorm2')(l2af)
        l2mp = MaxPooling1D()(l2af)

        l3 = Conv1D(32, 3,
                    padding='same',
                    kernel_initializer=initializer,
                    name='phoneme_Conv1D_3')(l2mp)
        # l3bn = BatchNormalization(epsilon=1e-05, momentum=0.1, name='BatchNorm3')(l3)
        l3af = sigmoid(l3)
        l3mp = MaxPooling1D()(l3af)

        l3fl = Flatten()(l3mp)

        out_eye = Dense(256,
                        kernel_initializer=initializer, name='out_eye', activation='linear')(l3fl)
        out_mouth = Dense(256,
                          kernel_initializer=initializer, name='out_mouth', activation='linear')(l3fl)

        model = Model(inputs=net_in, outputs=[out_mouth, out_eye])
        model.summary()
        opt = adam_v2.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss={'out_mouth': 'mse', 'out_eye': 'mse'},
                      loss_weights={'out_mouth': 1.0, 'out_eye': 1.0}, metrics=['accuracy'])
    else:
        model = load_model(load_model_dir)
        logging.info('Trained_model loaded')
    return model



def train_model(model, x_train, y_mouth_train, y_eye_train, x_test, y_mouth_test, y_eye_test, epochs, batch_size, log_dir, save_model_dir):
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    logging.info('Starting Wave2vec CNN model training')
    history = model.fit(x=x_train,
                        y=[y_mouth_train, y_eye_train],
                        validation_data=(x_test, [y_mouth_test, y_eye_test]),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[tbCallBack],
                        shuffle=True)
    # Plot model structure
    plot_model(model, to_file='model_w2v_params_cnn_plot.png', show_shapes=True, show_layer_names=True)
    # Log loss to console and file
    logging.info('Training loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['out_mouth_loss'][-1], history.history['out_eye_loss'][-1]))
    logging.info('Validation loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['val_out_mouth_loss'][-1], history.history['val_out_mouth_loss'][-1]))
    # Save model to file
    model.save(os.path.join(save_model_dir, 'w2v_cnn_epoch{}_bs{}.h5'.format(epochs, batch_size)))
    logging.info('Model saved to {}'.format(save_model_dir))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-input-data-dir', type=str, default='data/wave2vec/train_wave2vec_softmax.npy', help='path to training input data')
    parser.add_argument('--test-input-data-dir', type=str, default='data/wave2vec/test_wave2vec_softmax.npy', help='path to test input data')
    parser.add_argument('--train-eyes-data-dir', type=str, default='data/animation-params/eyes/train_scaled_eyes.npy', help='path to eye animation params training data')
    parser.add_argument('--test-eyes-data-dir', type=str, default='data/animation-params/eyes/test_scaled_eyes.npy', help='path to eye animation params test data')
    parser.add_argument('--train-mouth-data-dir', type=str, default='data/animation-params/mouth/train_scaled_mouth.npy', help='path to mouth animation params training data')
    parser.add_argument('--test-mouth-data-dir', type=str, default='data/animation-params/mouth/test_scaled_mouth.npy', help='path to mouth animation params test data')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--batch-size', type=int, default=16, help='batch size for training')
    parser.add_argument('--resume-training', type=bool, default=False, help='whether to resume training from a saved model')
    parser.add_argument('--save-model-dir', type=str, default='w2v/model/', help='directory where trained model will be saved')
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




