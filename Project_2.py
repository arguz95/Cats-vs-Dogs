"""
Created on Thu Oct 29 21:09:18 2020
Author: Arnaud Guzman-Annès
McGill University
"""

#pip install tensorflow
#pip install Keras

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

#image preprocessing -- Training
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
#image preprocessing -- Test
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('data/validation',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

############ CNN ############

#initializing
cnn = tf.keras.models.Sequential()

#Convolutional layer
cnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
'''
Change to  Conv layer to 1D or 2D and shape
'''
#Convo

#Pooling layer
cnn.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
'''
Change to  MaxPool layer to 1D or 2D
'''
#Convolutional and pool 2
cnn.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool1D(pool_size=2, strides=2))
'''
Change to Conv layer to 1D or 2D
Change to  MaxPool layer to 1D or 2D
'''

#Flatenning
cnn.add(tf.keras.layers.Flatten())

#Full conection  (ANN)
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#Output layer
#1 and sigmoid as our categories ara binary 
#units = number of neurons
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
'''
As we are trying to predict the churn (1 leaves 0 stays) we keep the sigmoid function
'''

#compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
'''
change metrics to MSE
'''

'''
Epochs = 25 
OR
Epochs = 15
'''

#training CNN and evaluation on test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
'''
we can start with 15 epochs, check accuracy and try higher number if needed)
As we have 10,000 rows; I guess it'll take some time to run...
'''

############ Testing ############

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('/Users/arnaudguzman-annes/Desktop/Arnaud/Etudes/MMA - McGill University/Courses/Fall 2020/INSY 662 - Data Mining and Visualization/Project 2/lechat.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices

#Result of the CNN
#[acces the batch][access the prediction]
if result[0][0] == 1:
  prediction = 'dog'
else:
  prediction = 'cat'
   
print(prediction)
