import numpy as np
import time
from scipy import ndimage
import os

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

#load CIFAR10 data
import h5py
CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
x_train = np.float32(CIFAR10_data['X_train'][:] )
y_train = np.int32(np.array(CIFAR10_data['Y_train'][:]))
print x_train.shape, y_train.shape
aug_X_train = []
aug_Y_train = []

#data augmentation
for xt in  x_train:
	temp_image = np.rollaxis(xt, 0, 3)
	temp_image_flr = np.fliplr(temp_image)  # flip
	temp_image_con = (temp_image - temp_image.min()) / (temp_image.max() - temp_image.min())  # normalize
	temp_image_med = ndimage.median_filter(temp_image, 2)  # median filter
	temp_image_flr = np.rollaxis(temp_image_flr, 2, 0)
    temp_image_con = np.rollaxis(temp_image_con, 2, 0)
    temp_image_med = np.rollaxis(temp_image_med, 2, 0)
    aug_X_train.append(xt)
	aug_X_train.append(temp_image_flr)
	aug_X_train.append(temp_image_con)
	aug_X_train.append(temp_image_med)

for yt in y_train:
	aug_Y_train.append(yt)
	aug_Y_train.append(yt)
	aug_Y_train.append(yt)
	aug_Y_train.append(yt)

print len(aug_X_train[:])
a_X_train =  np.float32(aug_X_train[:])
a_Y_train =  np.int32(aug_Y_train[:])
x_train = a_X_train
y_train = a_Y_train
print x_train.shape, y_train.shape
x_test = np.float32(CIFAR10_data['X_test'][:] )
y_test = np.int32( np.array(CIFAR10_data['Y_test'][:]  ) )

CIFAR10_data.close()


#D = 32
num_outputs = 10


class Conv_NN(Chain):
    def __init__(self):
        super(Conv_NN, self).__init__(
            conv1_1=L.Convolution2D(3, 64, 3, pad=1),
            bn1_1=L.BatchNormalization(64),
            conv1_2=L.Convolution2D(64, 64, 3, pad=1),
            bn1_2=L.BatchNormalization(64),

            conv2_1=L.Convolution2D(64, 128, 3, pad=1),
            bn2_1=L.BatchNormalization(128),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1),
            bn2_2=L.BatchNormalization(128),

            conv3_1=L.Convolution2D(128, 256, 3, pad=1),
            bn3_1=L.BatchNormalization(256),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1),
            bn3_2=L.BatchNormalization(256),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1),
            bn3_3=L.BatchNormalization(256),
            conv3_4=L.Convolution2D(256, 256, 3, pad=1),
            bn3_4=L.BatchNormalization(256),

            fc4 = L.Linear(4096, 500),
            fc5 = L.Linear(500, 500),
            fc6 = L.Linear(500,10),
        )
    def __call__(self, x_data, y_data, dropout_bool, bn_bool, p):
        x = Variable(x_data)
        t = Variable(y_data)
        h = F.relu(self.bn1_1(self.conv1_1(x), bn_bool))
        h = F.relu(self.bn1_2(self.conv1_2(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn2_1(self.conv2_1(h), bn_bool))
        h = F.relu(self.bn2_2(self.conv2_2(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn3_1(self.conv3_1(h), bn_bool))
        h = F.relu(self.bn3_2(self.conv3_2(h), bn_bool))
        h = F.relu(self.bn3_3(self.conv3_3(h), bn_bool))
        h = F.relu(self.bn3_4(self.conv3_4(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.dropout(F.relu(self.fc4(h)), p, dropout_bool)
        h = F.dropout(F.relu(self.fc5(h)), p, dropout_bool)
        h = self.fc6(h)
        L_out = h
        return F.softmax_cross_entropy(L_out, t), F.accuracy(L_out, t)

#returns test accuracy of the model.  dropout is set to its test state
def Calculate_Test_Accuracy(x_test, y_test, model, p, GPU_on, batch_size):
    L_Y_test = len(y_test)
    counter = 0
    test_accuracy_total = 0.0
    for i in range(0, L_Y_test, batch_size):
        if (GPU_on):
            x_batch = cuda.to_gpu(x_test[i:i+ batch_size,:])
            y_batch = cuda.to_gpu(y_test[i:i+ batch_size] )
        else:
            x_batch = x_test[i:i+batch_size,:]
            y_batch = y_test[i:i+batch_size]
        dropout_bool = False
        bn_bool = True
        loss, accuracy = model(x_batch, y_batch, dropout_bool,bn_bool, p)
        test_accuracy_batch  = 100.0*np.float(accuracy.data )
        test_accuracy_total += test_accuracy_batch
        counter += 1
    test_accuracy = test_accuracy_total/(np.float(counter))
    return test_accuracy


model =  Conv_NN()

#True if training with GPU, False if training with CPU
GPU_on = True

#size of minibatches
batch_size = 500

#transfer model to GPU
if (GPU_on):
    model.to_gpu()

#optimization method
optimizer = optimizers.Adam(alpha=0.001,beta1=0.9,beta2=0.999,eps=1e-08)
optimizer.setup(model)


#learning rate
optimizer.lr = .01

#dropout probability
p = .40

#number of training epochs
num_epochs = 400

L_Y_train = len(y_train)

time1 = time.time()
for epoch in range(num_epochs):
    #reshuffle dataset
    I_permutation = np.random.permutation(L_Y_train)
    x_train = x_train[I_permutation,:]
    y_train = y_train[I_permutation]
    epoch_accuracy = 0.0
    batch_counter = 0
    for i in range(0, L_Y_train, batch_size):
        if (GPU_on):
            x_batch = cuda.to_gpu(x_train[i:i+batch_size,:])
            y_batch = cuda.to_gpu(y_train[i:i+batch_size] )
        else:
            x_batch = x_train[i:i+batch_size,:]
            y_batch = y_train[i:i+batch_size]
        model.zerograds()
        dropout_bool = True
        bn_bool = False
        loss, accuracy = model(x_batch, y_batch, dropout_bool, bn_bool, p)
        loss.backward()
        optimizer.update()
        epoch_accuracy += np.float(accuracy.data)
        batch_counter += 1
    if (epoch % 1 == 0):
        train_accuracy = 100.0*epoch_accuracy/np.float(batch_counter)
        test_accuracy = Calculate_Test_Accuracy(x_test, y_test, model, p, GPU_on, batch_size)
	print 'Epoch', epoch, ',Train accuracy ', train_accuracy, ',Test accuracy', test_accuracy


time2 = time.time()
training_time = time2 - time1
print "Rank: %d" % rank
print "Training time: %f" % training_time


