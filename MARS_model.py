# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 14:59:44 2021

@author: xxx
"""

"""
import all the necessary packages
Tested with:
    
Tensorflow 2.2.0
Keras 2.3.0
Python 3.7

"""
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
from sklearn import metrics

from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout

# set the directory
import os
path = os.getcwd()
os.chdir(path)


#load the feature and labels, 24066, 8033, and 7984 frames for train, validate, and test
featuremap_train = np.load('feature/featuremap_train.npy')
featuremap_validate = np.load('feature/featuremap_validate.npy')
featuremap_test = np.load('feature/featuremap_test.npy')

labels_train = np.load('feature/labels_train.npy')
labels_validate = np.load('feature/labels_validate.npy')
labels_test = np.load('feature/labels_test.npy')

# Initialize the result array
paper_result_list = []


# define batch size and epochs
batch_size = 128
epochs = 150



#define the model
def define_CNN(in_shape, n_keypoints):


    in_one = Input(shape=in_shape)
    conv_one_1 = Conv2D(16, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding = 'same')(in_one)
    conv_one_1 = Dropout(0.3)(conv_one_1)
    conv_one_2 = Conv2D(32, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding = 'same')(conv_one_1)
    conv_one_2 = Dropout(0.3)(conv_one_2)

    
    conv_one_2 = BatchNormalization(momentum=0.95)(conv_one_2)


    fe = Flatten()(conv_one_2)
    # dense1
    dense_layer1 = Dense(512, activation='relu')(fe)
    dense_layer1 = BatchNormalization(momentum=0.95)(dense_layer1)
    # # dropout

    # dropout
    dense_layer1 = Dropout(0.4)(dense_layer1)
    
    out_layer = Dense(n_keypoints, activation = 'linear')(dense_layer1)
    

    # model
    model = Model(in_one, out_layer)
    opt = Adam(lr=0.001, beta_1=0.5)

    # compile the model
    model.compile(loss='mse', optimizer=opt, metrics=['mae', 'mse', 'mape', tf.keras.metrics.RootMeanSquaredError()])
    return model






# Repeat i iteration to get the average result
for i in range(10):
    # instantiate the model
    keypoint_model = define_CNN(featuremap_train[0].shape, 57)
    # initial maximum error 
    score_min = 10
    history = keypoint_model.fit(featuremap_train, labels_train,
                                 batch_size=batch_size, epochs=epochs, verbose=1, 
                                 validation_data=(featuremap_validate, labels_validate))
    # save and print the metrics
    score_train = keypoint_model.evaluate(featuremap_train, labels_train,verbose = 1)
    print('train MAPE = ', score_train[3])
    score_test = keypoint_model.evaluate(featuremap_test, labels_test,verbose = 1)
    print('test MAPE = ', score_test[3])
    result_test = keypoint_model.predict(featuremap_test)

    # Plot accuracy
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Xval'], loc='upper left')
    plt.show()
    
    # Plot loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Xval'], loc='upper left')
    plt.xlim([0,100])
    plt.ylim([0,0.1])
    plt.show()
    
    


    # error for each axis
    print("mae for x is",metrics.mean_absolute_error(labels_test[:,0:19], result_test[:,0:19]))
    print("mae for y is",metrics.mean_absolute_error(labels_test[:,19:38], result_test[:,19:38]))
    print("mae for z is",metrics.mean_absolute_error(labels_test[:,38:57], result_test[:,38:57]))
    
    # matrix transformation for the final all 19 points mae
    x_mae = metrics.mean_absolute_error(labels_test[:,0:19], result_test[:,0:19], multioutput = 'raw_values')
    y_mae = metrics.mean_absolute_error(labels_test[:,19:38], result_test[:,19:38], multioutput = 'raw_values')
    z_mae = metrics.mean_absolute_error(labels_test[:,38:57], result_test[:,38:57], multioutput = 'raw_values')
    
    all_19_points_mae = np.concatenate((x_mae, y_mae, z_mae)).reshape(3,19)
    avg_19_points_mae = np.mean(all_19_points_mae, axis = 0)
    avg_19_points_mae_xyz = np.mean(all_19_points_mae, axis = 1).reshape(1,3)

    all_19_points_mae_Transpose = all_19_points_mae.T
    
    # matrix transformation for the final all 19 points rmse
    x_rmse = metrics.mean_squared_error(labels_test[:,0:19], result_test[:,0:19], multioutput = 'raw_values', squared=False)
    y_rmse = metrics.mean_squared_error(labels_test[:,19:38], result_test[:,19:38], multioutput = 'raw_values', squared=False)
    z_rmse = metrics.mean_squared_error(labels_test[:,38:57], result_test[:,38:57], multioutput = 'raw_values', squared=False)
    
    all_19_points_rmse = np.concatenate((x_rmse, y_rmse, z_rmse)).reshape(3,19)
    avg_19_points_rmse = np.mean(all_19_points_rmse, axis = 0)
    avg_19_points_rmse_xyz = np.mean(all_19_points_rmse, axis = 1).reshape(1,3)

    all_19_points_rmse_Transpose = all_19_points_rmse.T
    
    # merge the mae and rmse
    all_19_points_maermse_Transpose = np.concatenate((all_19_points_mae_Transpose,all_19_points_rmse_Transpose), axis = 1)*100
    avg_19_points_maermse_Transpose = np.concatenate((avg_19_points_mae_xyz,avg_19_points_rmse_xyz), axis = 1)*100
    
    # concatenate the array, the final format is the same as shown in paper. First 19 rows each joint, the final row is the average
    paper_result_maermse = np.concatenate((all_19_points_maermse_Transpose, avg_19_points_maermse_Transpose), axis = 0)
    paper_result_maermse = np.around(paper_result_maermse, 2)
    # reorder the columns to make it xmae, xrmse, ymae, yrmse, zmae, zrmse, avgmae, avgrmse
    paper_result_maermse = paper_result_maermse[:, [0,3,1,4,2,5]]

    # append each iterations result
    paper_result_list.append(paper_result_maermse)
    
    #define the output directory
    output_direct = 'model/'
    
    if not os.path.exists(output_direct):
        os.makedirs(output_direct)

    # save the best model so far
    if(score_test[1] < score_min):
        keypoint_model.save(output_direct + 'MARS.h5')
        score_min = score_test[1]


# average the result for all iterations
mean_paper_result_list = np.mean(paper_result_list, axis = 0)
mean_mae = np.mean(np.dstack((mean_paper_result_list[:,0], mean_paper_result_list[:,2], mean_paper_result_list[:,4])).reshape(20,3), axis = 1)
mean_rmse = np.mean(np.dstack((mean_paper_result_list[:,1], mean_paper_result_list[:,3], mean_paper_result_list[:,5])).reshape(20,3), axis = 1)
mean_paper_result_list = np.concatenate((np.mean(paper_result_list, axis = 0), mean_mae.reshape(20,1), mean_rmse.reshape(20,1)), axis = 1)

#Export the Accuracy
output_path = output_direct + "Accuracy"
output_filename = output_path + "/MARS_accuracy"
if not os.path.exists(output_path):
    os.makedirs(output_path)
np.save(output_filename + ".npy", mean_paper_result_list)
np.savetxt(output_filename + ".txt", mean_paper_result_list,fmt='%.2f')
