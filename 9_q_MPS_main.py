import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import blocks
import time
from datetime import timedelta
import math
import os
from scipy.stats import ortho_group
from numpy import genfromtxt
from get_iris_data import *
import time
from sklearn import linear_model, datasets


# Taking iris data from sklearn
from sklearn import datasets
# 

from sklearn.model_selection import train_test_split
import sklearn

sess = tf.InteractiveSession()

model, qubits, thetas = blocks.mps_9_online()


def get_batch(bsize):
    batch = [0]*2
    r=np.random.randint(0,y_train.shape[0])
    batch[0] = np.squeeze(np.array([[np.cos(x_train[r])], [np.sin(x_train[r])]]))
    batch[1] = np.array([y_train[r]]).astype('float32')    
    return batch

def get_train_accuracy():
    train_accuracy = 0.0    
    for i in range(y_train.shape[0]):
        batch = [0]*2
        r=i
        batch[0] = np.squeeze(np.array([[np.cos(x_train[r])], [np.sin(x_train[r])]]))
        batch[1] = np.array([y_train[r]]).astype('float32')  
        train_accuracy = train_accuracy + accuracy.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3], 
                                          qubits[4]: batch[0][:,4], 
                                          qubits[5]: batch[0][:,5], 
                                          qubits[6]: batch[0][:,6], 
                                          qubits[7]: batch[0][:,7], 
                                          qubits[8]: batch[0][:,8], 

                                          y_truth: batch[1]})/float(y_train.shape[0])
    return train_accuracy


def get_test_accuracy():
    test_accuracy = 0.0
    for i in range(y_test.shape[0]):
        batch = [0]*2
        r=i
        batch[0] = np.squeeze(np.array([[np.cos(x_test[r])], [np.sin(x_test[r])]]))
        batch[1] = np.array([y_test[r]]).astype('float32')  
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3], 
                                          qubits[4]: batch[0][:,4], 
                                          qubits[5]: batch[0][:,5], 
                                          qubits[6]: batch[0][:,6], 
                                          qubits[7]: batch[0][:,7], 
                                          qubits[8]: batch[0][:,8], 
 
                                          y_truth: batch[1]})/float(y_test.shape[0])
    return test_accuracy


y_truth = tf.placeholder(tf.float32, shape=[1])

## Prep the data
# Let's trying building
#buildDataFromIris()
# Reading in the data
data = genfromtxt('data/sudoku_train.csv',delimiter=',')  # Training data
test_data = genfromtxt('data/sudoku_test.csv',delimiter=',')  # Test data

x_train=np.array([ i[1::] for i in data])
y_train,y_train_onehot = convertOneHot(data)




x_min_train = np.min(x_train,0)
x_train = x_train-x_min_train
x_max_train = np.max(x_train,0) # Normalize

# Doing a similiar conversion for the test data.
x_test=np.array([ i[1::] for i in test_data])
y_test,y_test_onehot = convertOneHot(test_data)

x_test = x_test-x_min_train

x_test = (x_test/x_max_train)
x_train = (x_train/x_max_train)

    ## Linear regression result
x_testLR = np.concatenate((np.ones((x_test.shape[0],1)),x_test),1)
x_trainLR = np.concatenate((np.ones((x_train.shape[0],1)),x_train),1)
#linear regression benchmark
#w=np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(x_trainLR),x_trainLR)),np.transpose(x_trainLR)),y_train)
#LRout=np.mean((np.dot(x_testLR,w)>0.5)==y_test)





#renorm:

x_test=x_test*np.pi/2#+ np.pi/4
x_train=x_train*np.pi/2#+ np.pi/4


#MPS SWITCH
x_test[:,2:5]=np.flip(x_test[:,2:5],1)
x_train[:,2:5]=np.flip(x_train[:,2:5],1)


error = np.abs(model - y_truth)
correct_prediction = tf.equal(model>0.5, y_truth>0.5)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.MomentumOptimizer(1e-3,0.9).minimize(error)

sess.run(tf.global_variables_initializer())


iters = 2000
bsize = 1
for i in range(iters):
    batch = get_batch(bsize)
    train_step.run(feed_dict={qubits[0]: batch[0][:,0], 
                              qubits[1]: batch[0][:,1], 
                              qubits[2]: batch[0][:,2], 
                              qubits[3]: batch[0][:,3],
                              qubits[4]: batch[0][:,4], 
                              qubits[5]: batch[0][:,5], 
                              qubits[6]: batch[0][:,6], 
                              qubits[7]: batch[0][:,7],
                              qubits[8]: batch[0][:,8], 
                              y_truth: batch[1]})
    if i % 10 == 0:
        train_accuracy = get_train_accuracy()
        print('step %d, training accuracy %g' % (i, train_accuracy))
        
        
print('Test acc is:' + str(get_test_accuracy()))


thetas_np=np.zeros(18)

thetas_np[0]=thetas[0][0].eval()
thetas_np[1]=thetas[0][1].eval()

thetas_np[2]=thetas[1][0].eval()
thetas_np[3]=thetas[1][1].eval()

thetas_np[4]=thetas[2][0].eval()
thetas_np[5]=thetas[2][1].eval()

thetas_np[6]=thetas[3][0].eval()
thetas_np[7]=thetas[3][1].eval()

thetas_np[8]=thetas[4][0].eval()
thetas_np[9]=thetas[4][1].eval()

thetas_np[10]=thetas[5][0].eval()
thetas_np[11]=thetas[5][1].eval()

thetas_np[12]=thetas[6][0].eval()
thetas_np[13]=thetas[6][1].eval()
"""
thetas_np[14]=thetas[7][0].eval()
thetas_np[15]=thetas[7][1].eval()

thetas_np[16]=thetas[8][0].eval()
thetas_np[17]=thetas[8][1].eval()

thetas_np[18]=thetas[9][0].eval()
thetas_np[19]=thetas[9][1].eval()

thetas_np[20]=thetas[10][0].eval()
thetas_np[21]=thetas[10][1].eval()
"""

thetas_np[14]=thetas[7][0].eval()
thetas_np[15]=thetas[7][1].eval()
thetas_np[16]=thetas[7][2].eval()
thetas_np[17]=thetas[7][3].eval()
"""
np.save('thetas_np_auto_iris.npy',thetas_np)


np.save('x_test_auto_iris_0_1.npy',x_test)
np.save('y_test_auto_iris_0_1.npy',y_test)"""
print(thetas_np)
