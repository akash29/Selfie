from data_loader import DataLoader
from resnet import ResNet
import numpy as np
import tensorflow as tf



def run_conv():
    session = tf.Session()
    dl = DataLoader("../Selfie-dataset")
    rs = ResNet()
    dl.load_data()
    training_data = dl.get_data(100)
    test_data = dl.get_data(100, is_train_data=False)

    iterator = tf.data.Iterator.from_structure(training_data.output_types,
                                       training_data.output_shapes)
    next_batch = iterator.get_next()

    #Make training and test data
    training_init_op = iterator.make_initializer(training_data)
    test_init_op = iterator.make_initializer(test_data)

    x = tf.placeholder(tf.float32, [100, 300, 300, 3])
    y = tf.placeholder(tf.float32, [100, 2])

    print("X ",x)
    print("Y ",y)

    session = tf.Session()

    for i in range(2):
        session.run(training_init_op)

        for j in range(2):
            img_batch, label_batch = session.run(next_batch)

            session.run(training_init_op, feed_dict={x: img_batch, y: label_batch})








"""
for epoch in range(num_epochs):


        # Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            # get next batch of data
            img_batch, label_batch = sess.run(next_batch)

            # And run the training op
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
"""



if __name__ == "__main__":
    run_conv()
