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
    x_train = np.load(train_input_data_dir).reshape(-1,8,13)
    x_test = np.load(test_input_data_dir).reshape(-1,8,13)
    y_eye_train = np.load(train_eyes_data_dir)
    y_eye_test = np.load(test_eyes_data_dir)
    y_mouth_train = np.load(train_mouth_data_dir)
    y_mouth_test = np.load(test_mouth_data_dir)
    return x_train, x_test, y_eye_train, y_eye_test, y_mouth_train, y_mouth_test

def define_model(x_train, timestep, resume_training, load_model_dir):
    if not resume_training:
        n_features = x_train.shape[2]
        lr = 1e-4
        initializer = 'glorot_uniform'

        # define trained_model
        net_in = Input(shape=(timestep, n_features))
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


def train_model(model, x_train, y_mouth_train, y_eye_train, x_test, y_mouth_test, y_eye_test, epochs, batch_size, log_dir, save_model_dir):
    tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
    logging.info('Starting VisemeID CNN model training')
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
    logging.info('Training loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['out_mouth_loss'][-1], history.history['out_eye_loss'][-1]))
    logging.info('Validation loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['val_out_mouth_loss'][-1], history.history['val_out_mouth_loss'][-1]))
    # Save model to file
    model.save(os.path.join(save_model_dir, 'visemeID_cnn_epoch{}_bs{}.h5'.format(epochs, batch_size)))
    logging.info('Model saved to {}'.format(save_model_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-input-data-dir', type=str, default='data/phones/train_viseme_ID_8timesteps.npy', help='path to training input data')
    parser.add_argument('--test-input-data-dir', type=str, default='data/phones/test_viseme_ID_8timesteps.npy', help='path to test input data')
    parser.add_argument('--train-eyes-data-dir', type=str, default='data/animation-params/eyes/train_scaled_eyes_8timesteps.npy', help='path to eye animation params training data')
    parser.add_argument('--test-eyes-data-dir', type=str, default='data/animation-params/eyes/test_scaled_eyes_8timesteps.npy', help='path to eye animation params test data')
    parser.add_argument('--train-mouth-data-dir', type=str, default='data/animation-params/mouth/train_scaled_mouth_8timesteps.npy', help='path to mouth animation params training data')
    parser.add_argument('--test-mouth-data-dir', type=str, default='data/animation-params/mouth/test_scaled_mouth_8timesteps.npy', help='path to mouth animation params test data')
    parser.add_argument('--timestep', type=int, default=8, help='number of timesteps')
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
    model = define_model(x_train, args.timestep, args.resume_training, args.load_model_dir)
    log_dir = os.path.join('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
    save_model_dir = args.save_model_dir
    train_model(model, x_train, y_mouth_train, y_eye_train, x_test, y_mouth_test, y_eye_test, args.epochs, args.batch_size, log_dir, save_model_dir)

if __name__ == '__main__':
    main()








