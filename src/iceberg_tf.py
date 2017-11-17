import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


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

print("Loading & Formatting Data")
train_df, train_images = load_and_format_data("../data/train.json")
test_df, test_images = load_and_format_data("../data/test.json")

# train_df = pd.read_json('../data/train.json')
# train_images = train_df.apply(lambda c_row: [np.stack([c_row['band_1'],c_row['band_2']], -1).reshape((75,75,2))],1)
# train_images = np.stack(train_images).squeeze()


# Train Test Split
# Splits training into training and validation data
# Test data is evaluated later in the code
print("Splitting Train/ Validation Data")
X_train, X_valid, y_train, y_valid = train_test_split(train_images, train_df['is_iceberg'].as_matrix(), test_size=0.2)

print('Train', X_train.shape, y_train.shape)
print('Validation', X_valid.shape, y_valid.shape)


# Convert data to float32 for use in TensorFlow
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_valid = X_valid.astype(np.float32)
y_valid= y_valid.astype(np.float32)


# Set the hyper parameters
learning_rate = 0.001
n_epochs = 10 # Changed from 2500
num_input = 75 * 75 # Size of the images
num_classes = 2 # Binary classifier
dropout = 0.4 # Dropout, probability to keep the units


# Design the CNN
# None used so the number of rows is flexible
# 2 used due to two channels of image data being passed in
X = tf.placeholder(tf.float32, shape=(None, 75, 75, 2), name="X")
y = tf.placeholder(tf.int64, shape=(None), name="y")


with tf.variable_scope("ConvNet"):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    conv1 = tf.layers.conv2d(X, filters=32, kernel_size=[5,5], activation = tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2,2], strides=2)

    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=[3,3], activation = tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2,2], strides=2)

    conv3 = tf.layers.conv2d(pool2, filters=128, kernel_size=[3,3], activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(pool3, filters=256, kernel_size=[3,3], activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(conv4, pool_size=[2, 2], strides=2)


    # Flatten data to 1D vector
    fc1 = tf.contrib.layers.flatten(pool4)

    fc2 = tf.layers.dense(fc1, 32, kernel_initializer=he_init, activation = tf.nn.relu)
    fc3 = tf.layers.dropout(fc2, rate=dropout)

    logits = tf.layers.dense(fc3, num_classes, activation=tf.nn.sigmoid)


# Define the loss function
with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")


# Define training method
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)


# Define evaluation
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


# Initialize the network
init = tf.global_variables_initializer()


# Train the model
print("Training CNN")
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(training_op, feed_dict={X: X_train, y:y_train})
        acc_train = accuracy.eval(feed_dict={X:X_train, y:y_train})
        acc_test = accuracy.eval(feed_dict={X:X_valid, y:y_valid})
        print(epoch, "Train_Acc:", acc_train, "Val_Acc:", acc_test)


# Transform the test data
test_images = test_images.astype(np.float32)
print("Test Images Shape:", test_images.shape)


# Make Predictions
# Z[:,1] used for second column, to finding probability of iceberg
# First column is probability of NOT being an iceberg
with tf.Session() as sess:
    Z = logits.eval(feed_dict={X:test_images})
    y_pred = Z[:,1]

output = pd.DataFrame(test_df['id'])
output['is_iceberg'] = y_pred

output.to_csv("cnn_submission.csv", index=False)
