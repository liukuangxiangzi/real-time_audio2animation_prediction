import os
import logging
import argparse
import numpy as np
from datetime import datetime
from keras.utils.vis_utils import plot_model
from keras.layers import Input, LSTM, LeakyReLU
from keras.models import Model
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.layers import Dropout
from keras.optimizers import adam_v2
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, Flatten, Conv1D


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
        n_features = x_train.shape[1]
        lr = 1e-4
        initializer = 'glorot_uniform'

        # define trained_model
        net_in = Input(shape=(n_features))
        l1 = Dense(128, kernel_initializer=initializer, name='l1')(net_in)
        lrelu1 = LeakyReLU(0.2)(l1)
        l2 = Dense(256, kernel_initializer=initializer, name='l2')(lrelu1)
        lrelu2 = LeakyReLU(0.2)(l2)
        l3 = Dense(512, kernel_initializer=initializer, name='l3')(lrelu2)
        lrelu2_2 = LeakyReLU(0.2)(l3)
        m1 = Dense(512, kernel_initializer=initializer, name='m1')(lrelu2_2)
        lrelu3 = LeakyReLU(0.2)(m1)
        m2 = Dense(256, kernel_initializer=initializer, name='m2')(lrelu3)
        lrelu4 = LeakyReLU(0.2)(m2)
        m3 = Dense(256, kernel_initializer=initializer, name='m_out', activation='linear')(lrelu4)
        e1 = Dense(512, kernel_initializer=initializer, name='e1')(lrelu2_2)
        elrelu3 = LeakyReLU(0.2)(e1)
        e2 = Dense(256, kernel_initializer=initializer, name='e2')(elrelu3)
        elrelu4 = LeakyReLU(0.2)(e2)
        e3 = Dense(256, kernel_initializer=initializer, name='e_out', activation='linear')(elrelu4)  #
        model = Model(inputs=net_in, outputs=[m3, e3])
        model.summary()
        opt = adam_v2.Adam(learning_rate=lr)
        model.compile(optimizer=opt, loss={'m_out': 'mse', 'e_out': 'mse'}, loss_weights={'m_out': 1.0, 'e_out': 1.0},
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
    plot_model(model, to_file='model_visemeID_params_mlp_plot.png', show_shapes=True, show_layer_names=True)
    # Log loss to console and file
    logging.info('Training loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['m_out_loss'][-1], history.history['e_out_loss'][-1]))
    logging.info('Validation loss: m_out={:.4f}, e_out={:.4f}'.format(history.history['val_m_out_loss'][-1], history.history['val_e_out_loss'][-1]))
    # Save model to file
    model.save(os.path.join(save_model_dir, 'visemeID_mlp_epoch{}_bs{}.h5'.format(epochs, batch_size)))
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




