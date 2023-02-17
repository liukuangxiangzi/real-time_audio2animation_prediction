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
from sklearn import decomposition



def pca_decomposition(input):
    pca = decomposition.PCA(n_components=32, whiten=True)
    pca.fit(input)
    input = pca.transform(input)
    # pca_basis_scale = np.sqrt(pca.explained_variance_)
    # input = np.divide(input, pca_basis_scale)
    return input

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
        # l1bn = BatchNormalization(epsilon=1e-05, momentum=0.1, name='BatchNorm1')(l1)
        l1af = sigmoid(l1)
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


def train_model(model, x_train, y_mouth_train, y_eye_train, x_test, y_mouth_test, y_eye_test, epochs, batch_size, save_model_dir):
    logging.info('Starting Wave2vec CNN_previous_params model training')

    batch_per_epoch = int(x_train.shape[0] / batch_size)
    batch_per_epoch_val = int(x_test.shape[0] / batch_size)
    for epoch in range(0, epochs):
        for j in range(batch_per_epoch):
            batch_x = x_train[j * batch_size:(j + 1) * batch_size]
            batch_y_e = y_eye_train[j * batch_size:(j + 1) * batch_size]
            batch_y_m = y_mouth_train[j * batch_size:(j + 1) * batch_size]
            if j == 0:
                loss, loss_m, loss_e, acc_m, acc_e = model.train_on_batch(batch_x, [batch_y_m, batch_y_e])
            else:
                last_y_m, last_y_e = model.predict(x_train[(j - 1) * batch_size:j * batch_size])
                last_y_concat = np.concatenate((last_y_m, last_y_e), axis=-1)
                last_y_pca = pca_decomposition(last_y_concat)
                batch_x[:, 0, :] = last_y_pca[:, :]
                loss, loss_m, loss_e, acc_m, acc_e = model.train_on_batch(batch_x, [batch_y_m, batch_y_e])
            print('>%d, %d/%d, loss=%.3f, loss_m=%.3f, loss_e=%.3f, acc_m=%.3f, acc_e=%.3f' % (
            epoch + 1, j + 1, batch_per_epoch, loss, loss_m, loss_e, acc_m, acc_e))
        for k in range(batch_per_epoch_val):
            batch_x_val = x_test[k * batch_size:(k + 1) * batch_size]
            batch_y_e_val = y_eye_test[k * batch_size:(k + 1) * batch_size]
            batch_y_m_val = y_mouth_test[k * batch_size:(k + 1) * batch_size]
            if k == 0:
                loss_val, loss_m_val, loss_e_val, acc_m_val, acc_e_val = model.train_on_batch(batch_x_val, [batch_y_m_val, batch_y_e_val])
            else:
                last_y_m_val, last_y_e_val = model.predict(x_test[(k - 1) * batch_size:k * batch_size])
                last_y_concat_val = np.concatenate((last_y_m_val, last_y_e_val), axis=-1)
                last_y_pca_val = pca_decomposition(last_y_concat_val)
                batch_x_val[:, 0, :] = last_y_pca_val[:, :]
                loss_val, loss_m_val, loss_e_val, acc_m_val, acc_e_val = model.train_on_batch(batch_x_val, [batch_y_m_val, batch_y_e_val])
            print('>%d, %d/%d, val_loss=%.3f, val_loss_m=%.3f, val_loss_e=%.3f, val_acc_m=%.3f, val_acc_e=%.3f' % (
                epoch + 1, k + 1, batch_per_epoch_val, loss_val, loss_m_val, loss_e_val, acc_m_val, acc_e_val))



    plot_model(model, to_file='model_w2v_params_cnn_previousParams_plot.png', show_shapes=True, show_layer_names=True)
    # Save model to file
    model.save(os.path.join(save_model_dir, 'w2v_cnn_previousParams_epoch{}_bs{}.h5'.format(epochs, batch_size)))
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
    parser.add_argument('--batch-size', type=int, default=32, help='batch size for training')
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
    #log_dir = os.path.join('logs', 'fit', datetime.now().strftime('%Y%m%d-%H%M%S'))
    save_model_dir = args.save_model_dir
    train_model(model, x_train, y_mouth_train, y_eye_train, x_test, y_mouth_test, y_eye_test, args.epochs, args.batch_size, save_model_dir)

if __name__ == '__main__':
    main()




#
#     val_loss = []
#     for batch, (X, y) in enumerate(test_data):
#         val_loss.append(validate_on_batch(X, y))
#
#     print('Validation Loss: ' + str(np.mean(val_loss)))






