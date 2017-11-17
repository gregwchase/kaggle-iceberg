import pandas as pd
import numpy as np
import mxnet as mx
from sklearn.model_selection import train_test_split
import time
import logging
from keras.utils import normalize

logging.getLogger().setLevel(logging.DEBUG)

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


start = time.time()

print("Loading & Formatting Data")
train_df, train_images = load_and_format_data("../data/train.json")
test_df, test_images = load_and_format_data("../data/test.json")


print("Splitting Train/ Validation Data")
X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_df['is_iceberg'].as_matrix(), test_size=0.2)

print('Train:', X_train.shape, y_train.shape)
print('Validation:', X_valid.shape, y_valid.shape)
print('Test:', test_images.shape)

X_train = X_train.reshape(X_train.shape[0], 2, 75, 75)
X_valid = X_valid.reshape(X_valid.shape[0], 2, 75, 75)

X_train = normalize(X_train, axis=-1)
X_valid = normalize(X_valid, axis=-1)


data = mx.sym.var('data')

# First Conv Layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=32)
relu = mx.sym.Activation(data=conv1, act_type="relu")

conv = mx.sym.Convolution(data=relu, kernel=(5,5), num_filter=32)
relu1 = mx.sym.Activation(data=conv, act_type="relu")

pool1 = mx.sym.Pooling(data=relu1, pool_type="max", kernel=(2,2), stride=(1,1))

# Second Conv Layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(2,2), num_filter=64)
relu2 = mx.sym.Activation(data=conv2, act_type="relu")
pool2 = mx.sym.Pooling(data=relu2, pool_type="max", kernel=(2,2), stride=(1,1))

# First Fully Connected Layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden = 500)
relu3 = mx.sym.Activation(data=fc1, act_type="relu")

# Second Fully Connected Layer
fc2 = mx.sym.FullyConnected(data=relu3, num_hidden=2)

# Softmax Loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

'''
Line 81: GPU context (for AWS)
Line 82: CPU context (for local, CPU training)
'''
# model = mx.mod.Module(symbol=lenet, context=mx.gpu(0))
model = mx.mod.Module(symbol=lenet, context=mx.cpu())


# Train Model

print("Creating MXNet NDArrayIter objets")
batch_size = 50
n_epochs = 10
train_iter = mx.io.NDArrayIter(X_train, y_train, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(X_valid, y_valid, batch_size)


print("Training Model")
model.fit(train_iter, eval_data = val_iter, optimizer='adam',
            optimizer_params={'learning_rate': 0.01},
            eval_metric='acc',
            batch_end_callback = mx.callback.Speedometer(batch_size, 100),
            num_epoch=n_epochs)

test_images = test_images.reshape(test_images.shape[0], 2, 75, 75)
test_iter = mx.io.NDArrayIter(test_images, None, batch_size)

# Predict against the test data
predictions = model.predict(test_iter)
predictions = predictions[:,1] # Column indicating probability of an iceberg
predictions = predictions.asnumpy()
predictions = predictions.reshape(-1,1)

# Write output to CSV for submission
output = pd.DataFrame(test_df['id'])
output['is_iceberg'] = predictions
output.to_csv('iceberg_submission.csv', index=False)

print("Seconds:", (time.time() - start))
