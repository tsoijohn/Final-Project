import numpy as np
import tensorflow as tf
from read_data import *
from matplotlib import pyplot as plt
from hoeffding import *

def svm(train_x,train_y,test_x,test_y):
    # Get shape
    num_data,num_feat = train_x.shape
    print("\nSVM Outputs")
    print("========================================================")
    #initialize weights and bias
    w = tf.Variable(tf.random_normal(shape=[num_feat,1]),dtype=tf.float32)
    b = tf.Variable(tf.random_normal(shape=[1,1]),dtype=tf.float32)

    #Initialize placeholder
    # x has dimensions of 3 with 3 parameters
    x = tf.placeholder(shape=[None, num_feat],dtype=tf.float32)
    # y is labelling dimension of 1
    y = tf.placeholder(shape=[None,1],dtype=tf.float32)

    # the dot product of x and w with b
    dp = tf.matmul(x,w) + b

    # Minimizing hinge loss
    # Reguarlization term lambda > 0
    lam = 0.1
    # Calculating hinge loss for SVM
    hinge_loss = tf.reduce_sum(tf.maximum(0.,1.-y*dp))
    # Reguarlized loss minimization
    reg_loss = (lam/2)*tf.reduce_sum(tf.square(w))
    svm_loss = reg_loss + hinge_loss

    # SGD as optimizer
    sol = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

    # Accuracy
    predicted_class = tf.sign(dp)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y,predicted_class),"float"))

    # Declare batch sizes
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch = 100
    iter = 500
    loss_vec = []
    train_accuracy = []
    test_accuracy = []
    train_loss = []
    test_loss = []
    for i in range(iter + 1):
        rand_index = np.random.choice(num_data, size=batch)
        X = train_x[rand_index]
        Y = np.transpose([train_y[rand_index]])
        sess.run(sol, feed_dict={x: X, y: Y})
        temp_loss = sess.run(svm_loss, feed_dict={x:X,y:Y})
        loss_vec.append(temp_loss)
        temp_train_loss = sess.run(svm_loss, feed_dict={x:train_x, y:np.transpose([train_y])})
        train_loss.append(temp_train_loss)
        temp_testloss = sess.run(svm_loss, feed_dict={x:test_x,y:np.transpose([test_y])})
        test_loss.append(temp_testloss)
        train_acc_temp = sess.run(accuracy, feed_dict={x: train_x, y: np.transpose([train_y])})
        train_accuracy.append(train_acc_temp)
        test_acc_temp = sess.run(accuracy, feed_dict={x: test_x, y: np.transpose([test_y])})
        test_accuracy.append(test_acc_temp)
        
        if (i % 100 == 0):
            print("iteration", i, "loss", loss_vec[i])
            print("iteration",i,"train accuracy", train_accuracy[i])
            print("iteration",i,"test accuracy", test_accuracy[i])
    
    # Plot train/test accuracies
    plt.plot(train_accuracy, 'g-', label='Training Accuracy')
    plt.plot(test_accuracy, 'r--', label='Test Accuracy')
    plt.title('Train and Test Set Accuracies')
    plt.savefig('accuracy.png')
    plt.show()
    
    # Plot loss over time
    plt.plot(train_loss,'g-')
    plt.plot(test_loss,'r--')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.show()

    # Hoeffding's
    erisk = 1- sum(test_accuracy)/len(test_accuracy)
    CI_L,CI_U = hoeffding(erisk,0.05)
    print("\nHoeffding's Confidence interval for SVM is:")
    print(CI_L,CI_U)

    return(CI_L,CI_U)