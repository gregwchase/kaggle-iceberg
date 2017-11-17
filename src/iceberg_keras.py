import os

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
import time

np.random.seed(1337)


def load_and_format_data(file_path):
    """
    Loads and formats the JSON data into NumPy arrays

    INPUT
        file_path: file path to the JSON file.

    OUTPUT
        Pandas DataFrame
        NumPy array of arrays
    """

    df = pd.read_json(file_path)
    images = df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
    images = np.stack(images).squeeze()
    return df, images



def reshape_data(arr, img_rows, img_cols, channels):
    """
    Reshapes the data into format for CNN.

    INPUT
        arr: Array of NumPy arrays.
        img_rows: Image height
        img_cols: Image width
        channels: Specify if the image is grayscale (1) or RGB (3)

    OUTPUT
        Reshaped array of NumPy arrays.
    """
    return arr.reshape(arr.shape[0], img_rows, img_cols, channels)


def cnn_model(X_train, X_test, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes):
    """
    Define and run the Convolutional Neural Network

    INPUT
        X_train: Array of NumPy arrays
        X_test: Array of NumPy arrays
        y_train: Array of labels
        y_test: Array of labels
        kernel_size: Initial size of kernel
        nb_filters: Initial number of filters
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification

    OUTPUT
        Fitted CNN model
    """

    model = Sequential()

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.20))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    stop = EarlyStopping(monitor='val_acc',
                         min_delta=0.001,
                         patience=1,
                         verbose=0,
                         mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1,
              validation_split=0.2,
              class_weight='auto',
              callbacks=[stop, tensor_board])

    return model


if __name__ == '__main__':
    start = time.time()

    print("Loading & Formatting Data")
    train_df, X_train = load_and_format_data("../data/train.json")
    test_df, X_test = load_and_format_data("../data/test.json")


    print("Splitting Train/ Validation Data")
    y_train = train_df['is_iceberg'].as_matrix()

    print('Train:', X_train.shape, y_train.shape)
    print('Test:', X_test.shape)


    # Specify parameters before model is run.
    batch_size = 100
    nb_classes = 2
    nb_epoch = 20

    img_rows, img_cols = 75, 75
    channels = 2
    nb_filters = 32
    kernel_size = (5,5)
    input_shape = (img_rows, img_cols, channels)


    print("Reshaping Data")
    X_train = reshape_data(X_train, img_rows, img_cols, channels)
    X_test = reshape_data(X_test, img_rows, img_cols, channels)


    print("Normalizing Data")
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255


    y_train = np_utils.to_categorical(y_train, nb_classes)
    print("y_train Shape: ", y_train.shape)


    print("Training Model")
    model = cnn_model(X_train, X_test, y_train, kernel_size, nb_filters, channels, nb_epoch, batch_size,
                      nb_classes)

    print("Predicting")
    y_pred = model.predict(X_test)
    y_pred = y_pred[:,1].reshape(-1,1) # probability of being an iceberg


    print("Writing Submission")
    output = pd.DataFrame(test_df['id'])
    output['is_iceberg'] = y_pred
    output.to_csv('keras_submission.csv', index=False)

    print("Seconds:", (time.time() - start))
