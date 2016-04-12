import os
import sys
import numpy as np
import time
import math
import cv2
from datetime import datetime
import csv
from itertools import islice

import tensorflow as tf
import tensorflow.python.platform

# Parameters
display_step = 10
batch_size = 32
resize_w = 28
resize_h = 28
n_classes = 10 # total classes (0-9 digits)
n_char_to_predict = 1
dropout = 1 # Dropout, probability to keep units
n_epochs = 1
learning_rate = 0.0001
n_channels = 1 # number of channels in the input image

def conv2d(input_op, n_out_fmap, kh, kw, k):
    # Initialize the convolution parameters
    n_in_fmap = input_op.get_shape()[-1].value
    init_range = math.sqrt(6.0 / (kh*kw*n_in_fmap + n_out_fmap*kh*kw/k))
    w = tf.Variable(tf.random_uniform([kh, kw , n_in_fmap, n_out_fmap], minval=-init_range, maxval=init_range))
    b = tf.Variable(tf.zeros([n_out_fmap]))

    # perform convolution and apply relu activation function 
    activation = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_op, w, strides=[1, 1, 1, 1], padding='SAME'),b))

    return activation

def max_pool(input_op, k):
    return tf.nn.max_pool(input_op, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def dense(input_op, n_out, ACTIVATION="RELU"):

    # Initialize the dense layer parameters
    n_in = input_op.get_shape()[-1].value
    init_range = math.sqrt(6.0 / (n_in + n_out))
    w = tf.Variable(tf.random_uniform([n_in, n_out], minval=-init_range, maxval=init_range))
    b = tf.Variable(tf.zeros([n_out]))

    if ACTIVATION == "RELU":
        activation = tf.nn.relu(tf.add(tf.matmul(input_op, w), b)) # Relu activation
    else:
        activation = tf.add(tf.matmul(input_op, w), b) # linear activation

    return activation


def conv_net(X, dropout):
    n_fmaps = [32, 64, 192]
    kernel_size = [(5,5), (5,5), (3,3)]
    max_pool_strides = [1,2,2]
    fc_layers = [4096, 4096]

    # Convolution Layer 1
    conv1 = conv2d(X, n_out_fmap=n_fmaps[0], kh=kernel_size[0][0], kw=kernel_size[0][1], k=max_pool_strides[0])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=max_pool_strides[0])
    # Local response normalization (LRN)
    conv1 = tf.nn.local_response_normalization(conv1)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, dropout)

    # Convolution Layer 2
    conv2 = conv2d(conv1, n_out_fmap=n_fmaps[1], kh=kernel_size[1][0], kw=kernel_size[1][1], k=max_pool_strides[1])
    # Local response normalization (LRN)
    conv2 = tf.nn.local_response_normalization(conv2)
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=max_pool_strides[1])
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, dropout)

    # Convolution Layer 3
    conv3 = conv2d(conv2, n_out_fmap=n_fmaps[2], kh=kernel_size[2][0], kw=kernel_size[2][1], k=max_pool_strides[2])
    # Local response normalization (LRN)
    conv3 = tf.nn.local_response_normalization(conv3)
    # Max Pooling (down-sampling)
    conv3 = max_pool(conv3, k=max_pool_strides[2])
    # Apply Dropout
    conv3 = tf.nn.dropout(conv3, dropout)

    # Fully connected layers
    shp = conv3.get_shape()
    flattened_shape = shp[1].value*shp[2].value*shp[3].value
    dense1 = tf.reshape(conv3, [-1,flattened_shape])
    dense1 = dense(dense1, n_out=fc_layers[0], ACTIVATION="RELU")
    dense1 = tf.nn.dropout(dense1, dropout) # Apply Dropout

    dense2 = dense(dense1, n_out=fc_layers[1], ACTIVATION="RELU")
    dense2 = tf.nn.dropout(dense2, dropout) # Apply Dropout

    # Output (for class prediction)
    out = dense(dense1, n_out=n_classes, ACTIVATION=None)

    return out

def loss_function(pred, y):
    """
    calculates loss function using categorical cross_entropy
    """
    y = tf.expand_dims(y, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat(1, [indices, y])
    onehot_labels = tf.sparse_to_dense(concated, tf.pack([batch_size, n_classes]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(pred, onehot_labels)
    
    cost = tf.reduce_mean(cross_entropy)

    return cost


def train():
    with tf.Graph().as_default():
        # tf Graph input
        x = tf.placeholder("float", [batch_size, resize_h, resize_w, 1])
        y = tf.placeholder(tf.int32, [batch_size]) 
        keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # Construct model
        pred = conv_net(x, keep_prob)

        # Define loss and optimizer
        cost = loss_function(pred, y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.cast(tf.argmax(pred,1), tf.int32), y)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        tf.scalar_summary("loss_value", cost)
        summary_op = tf.merge_all_summaries()

        # Create a saver
        saver = tf.train.Saver(tf.all_variables())

        # Initializing the variables
        init = tf.initialize_all_variables()

        # Launch the graph
        with tf.Session() as sess:
            with tf.device("/gpu:0"):
                sess.run(init)
                writer = tf.train.SummaryWriter("train_logs", graph_def=sess.graph_def)
                tf.train.write_graph(sess.graph_def, './train_logs', 'train.pbtxt')

                print " Session initialized"
                
                """
                print " Loading the model..."
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)        
                print " Model loaded"
                """

                for epoch in range(n_epochs):
                    step = 0
                    try:
                        # Keep training until reach max iterations
                        while step < num_train_batches:

                                batch_xs, batch_y = get_batch_data(step)#mnist.train.next_batch(batch_size)
                                batch_y = np.array(batch_y, dtype=np.int32)

                                #print  " batch data generated"

                                # Fit training using batch data
                                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_y, keep_prob: dropout})

                                if step % display_step == 0:
                                    # Calculate batch accuracy
                                    acc = sess.run([accuracy, cost], feed_dict={x: batch_xs, y: batch_y, keep_prob: 0.75})
                                    # Calculate batch loss
                                    #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_y, keep_prob: 1.})
                                    print "Iter " + str(step) + ", Minibatch Loss= " + "{:.6f}".format(acc[1]) + ", Training Accuracy= " + "{:.5f}".format(acc[0])
                                step += 1

                                # Save the model after every 400 steps
                                if step%400 == 0:
                                    model_name = "model_%s_%s.ckpt"%(epoch, step)
                                    if not os.path.exists("checkpoints"):
                                        os.mkdir("checkpoints")

                                    checkpoint_path = saver.save(sess, os.path.join("checkpoints",model_name))
                                    print("saving model %s" % checkpoint_path)                

                    except Exception as e:
                        print " Training aborted at batch index %s"%(step)
                        print e
                
                # Calculate output for test images
                try:
                    total_predictions = []
                    for tst_idx in xrange(num_test_batches):
                        batch_xs, batch_y = get_batch_data(tst_idx, mode = "TEST")#mnist.train.next_batch(batch_size)
                        #batch_y = np.array(batch_y, dtype=np.int32)
                        #print "batch data generated"
                        predictions = sess.run(pred, feed_dict={x: batch_xs, keep_prob: 1.})
                        predictions = np.argmax(predictions,1)
                        #print  predictions
                        total_predictions = np.append(total_predictions, predictions)

                    with  open("submission.csv",'w') as f:
                        fieldnames = ['ImageId', 'Label']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for i,j in enumerate(total_predictions):
                            writer.writerow({'ImageId':i+1, 'Label':int(j)})
                except:
                    print 
                    print "Testing aborted"


def get_batch_data(batch_id, mode="TRAIN"):
    """ read the extra data structure and load the images based on batch id

    Args: 
    batch_id: integer
    mode: "TRAIN" or "TEST"

    Returns:
    A tuple (batch_images, batch_labels) 

    """
    start = batch_id*batch_size + 1   
    end = (batch_id + 1)*batch_size + 1
    batch_images = []
    batch_labels = []

    if mode == "TRAIN":
        try:
            with open(train_data_path, 'r') as f:
                reader = csv.reader(f)
                for row in islice(reader, start, end):
                    row = map(int, row)
                    row = np.array(row)
                    batch_labels.append(row[0])
                    batch_images.append(row[1:].reshape((resize_h, resize_w,1)))

                #print batch_labels
        except Exception as e:
            print "Unable to read the CSV file from location start- %s, end- %s"%(start, end)
    else:
        try:
            with open(test_data_path, 'r') as f:
                reader = csv.reader(f)
                for row in islice(reader, start, end):
                    row = map(int, row)
                    row = np.array(row)
                    batch_images.append(row.reshape((resize_h, resize_w,1)))

                #print batch_labels
        except Exception as e:
            print "Unable to read the CSV file from location start- %s, end- %s"%(start, end)        

    try:
        batch_images = np.array(batch_images, dtype=np.float32)

    except Exception as e:
        print "Unable to reshape the batch images from location start- %s, end- %s"%(start, end)

    return (batch_images, batch_labels)


if __name__ == '__main__':

    root = './'
    dataset_path = "./"
    train_data_path = os.path.join(dataset_path,"train.csv")
    test_data_path = os.path.join(dataset_path,"test.csv")

    checkpoint_dir = "./checkpoints/"
    # read the train csv file
    with open(train_data_path,'r') as f:
        reader =  csv.reader(f)
        num_train_files = sum(1 for l in reader)

    with open(test_data_path,'r') as f:
        reader =  csv.reader(f)
        num_test_files = sum(1 for l in reader)

    # as the first row contains the name of the columns
    num_train_files = num_train_files - 1
    num_test_files = num_test_files - 1

    num_train_batches = int(num_train_files/batch_size)
    num_test_batches = int(num_test_files/batch_size)
    
    print num_test_batches, num_test_files
    
    train()
    
