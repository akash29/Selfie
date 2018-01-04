from data_loader import DataLoader
from resnet import ResNet
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def run_conv(learning_rate=0.009, num_epochs=100, batch_size=100, print_cost=True):

    dl = DataLoader("../Selfie-dataset")
    rs = ResNet()
    dl.load_data()

    train_batches = int(dl.num_train/batch_size)

    test_batches = int(dl.num_test/batch_size)

    training_data = dl.get_data(batch_size)
    test_data = dl.get_data(batch_size, is_train_data=False)

    iterator = tf.data.Iterator.from_structure(training_data.output_types,
                                       training_data.output_shapes)
    next_batch = iterator.get_next()

    #Make training and test data
    training_init_op = iterator.make_initializer(training_data)
    test_init_op = iterator.make_initializer(test_data)

    X = tf.placeholder(tf.float32, [batch_size, 300, 300, 3])
    Y = tf.placeholder(tf.float32, [batch_size, 2])

    costs = []

    Z = rs.resNet_50(X)

    cost = rs.compute_cost(Z,Y)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    predict_op = tf.argmax(Z, 1)

    correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for i in range(num_epochs):
            sess.run(training_init_op)
            batch_cost = 0
            train_acc = 0.0
            for j in range(train_batches):
                img_batch, label_batch = sess.run(next_batch)
                _, temp_cost = sess.run([optimizer,cost],feed_dict={X:img_batch,Y:label_batch})
                batch_cost += temp_cost/train_batches
                train_acc += accuracy.eval({X:img_batch,Y:label_batch})

            train_acc /= train_batches
            print ("Training accuracy: ", train_acc)

            if print_cost and i % 5 ==0:
                print ("Epoch count={} and cost={}".format(i,batch_cost))
            if print_cost:
                costs.append(batch_cost)

            sess.run(test_init_op)
            test_acc = 0.0
            for k in range(test_batches):
                img_batch, label_batch = sess.run(next_batch)
                test_acc += sess.run(accuracy,feed_dict={X:img_batch,Y:label_batch})

            test_acc /= test_batches

            print("Test accuracy: ", test_acc)

        save_path = saver.save(sess,'/model1.ckpt')
        print ("Model save in{}".format(save_path))
        plt.plot(np.squeeze(costs))
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.title("cost characteristics with learning rate {}".format(learning_rate))
        plt.savefig("Cost characteristics-Learning rate {}".format(learning_rate))
        #plt.show()




if __name__ == "__main__":
    run_conv()
