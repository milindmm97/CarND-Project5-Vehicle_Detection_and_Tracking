# -*- coding: utf-8 -*-
"""
Created on Sun Mar 01 15:51:19 2018

@author: Milind
"""

from keras.layers import Conv2D, Flatten, Lambda, MaxPooling2D,Dense
from keras.models import Model, Sequential
import helper as aux
import glob
from sklearn.model_selection import train_test_split as trainTestSplit
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.preprocessing import image
import cv2
#from keras.layers import Dropout 
#from keras.preprocessing.image import ImageDataGenerator
#prepare the data

vehicleFolder = 'samples/vehicles/'
nonVehiclesFolder = 'samples/non-vehicles/'

#Search the directory and also search the subfolders 
vehicleFiles = glob.glob('{}*/*.png'.format(vehicleFolder), recursive=True)
nonVehicleFiles = glob.glob('{}*/*.png'.format(nonVehiclesFolder), recursive=True)
imageSamplesFiles = vehicleFiles + nonVehicleFiles
y = np.concatenate((np.ones(len(vehicleFiles)), np.zeros(len(nonVehicleFiles))))

 # Use skLearn utils to split data to train and test sets
xTrain, xTest, yTrain, yTest = trainTestSplit(imageSamplesFiles, y, test_size=0.2, random_state=12)
 # Further split train data to train and validation
xTrain, xVal, yTrain, yVal = trainTestSplit(xTrain, yTrain, test_size=0.2, random_state=12)

data = {'xTrain': xTrain, 'xValidation': xVal, 'xTest': xTest,
                        'yTrain': yTrain, 'yValidation': yVal, 'yTest': yTest}



#Start CNN build
def train(inputShape=(64, 64, 3)):

 inputShape=(64, 64, 3) #input size of image (since rgb -> 3)

 model = Sequential()

# Center and normalize our data
 model.add(Lambda(lambda x: x / 255., input_shape=inputShape, output_shape=inputShape))

# Block 0
 model.add(Conv2D(filters=16, kernel_size=(4, 4), activation='relu', name='cv0',
                     input_shape=inputShape, padding="same"))
# model.add(Dropout(0.5)) #Dropout to reduce overfitting


# Block 1 
#model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', name='cv1', padding="same"))
#model.add(Dropout(0.5))

# block 2
#model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', name='cv2', padding="same"))

 '''Max Pooling'''
 model.add(MaxPooling2D(pool_size=(8, 8))) # Stride value default to pool size
# model.add(Dropout(0.5))
 
 model.add(Flatten())
 
 model.add(Dense(output_dim=128,activation='relu')) #alternative1
 model.add(Dense(output_dim=1,activation='sigmoid')) #alternative2

# binary 'classifier' Fully connected layer 
#model.add(Conv2D(filters=1, kernel_size=(8, 8), name='fcn', activation="sigmoid"))
 
 return model, 'trained'

def main():

    trainSamples = aux.createSamples(xTrain, yTrain)
    validationSamples = aux.createSamples(xVal, yVal)

    sourceModel, modelName = train()

    x = sourceModel.output
#x = Flatten()(x) #comment if using CNN
    model = Model(inputs=sourceModel.input, outputs=x)

    print(model.summary())

#loss is 'mse' or 'binary_crossentropy'

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


    useFlips= True
    batchSize = aux.promptForInt(message='Please specify the batch size (32, 64, etc.): ')
    epochCount = aux.promptForInt(message='Please specify the number of epochs: ')

    trainGen =aux. generator(samples=trainSamples,batchSize=batchSize) #,useFlips=useFlips)
    validGen = aux.generator(samples=validationSamples,batchSize=batchSize)#, useFlips=useFlips)


#timeStamp = aux.timeStamp()
    weightsFile = '{}.h5'.format(modelName)

    checkpointer = ModelCheckpoint(filepath=weightsFile,
                                       monitor='val_acc', verbose=0, save_best_only=True)

    inflateFactor = 2 if useFlips else 1

    # Keras generator params computation
    stepsPerEpoch = len(trainSamples) * inflateFactor / batchSize
    validationSteps = len(validationSamples) * inflateFactor / batchSize
    model.fit_generator(trainGen,
                                steps_per_epoch=stepsPerEpoch,
                                validation_data=validGen,
                                validation_steps=validationSteps,
                                epochs=epochCount, callbacks=[checkpointer])

    print('Training complete. Weights for best validation accuracy have been saved to {}.'
              .format(weightsFile))
   # Evaluating accuracy on test set

    print('Evaluating accuracy on test set.')

# test for useFlips in testSamples without training
    testSamples = aux.createSamples(xTest, yTest)
    testGen = aux.generator(samples=testSamples, useFlips=True)
    testSteps = len(testSamples) / batchSize
    accuracy = model.evaluate_generator(generator=testGen, steps=testSteps)



    print('test accuracy: ', accuracy)

#Make preditction
'''
test1= 'cars_test/10.jpg'
test2 ='non_car_test/7.jpg'
image_test= cv2.imread(test2)
cv2.imshow('test',image_test)
cv2.waitKey(0)
cv2.destroyAllWindows()

test_image = image.load_img(test2,target_size = (64,64))
test_image= image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0) #since predict only works on
# four dimension


result =model.predict(test_image)

res =result[0][0]

if (result[0][0] >  0.5):
    print ('Vehcile found with a probablity of',res)
else:
   print('No vehcile found',res)
    

#calculation of confusion matrix

true_positive=0
false_negative=0

VehicleFiles_test = glob.glob('*.ppm')
len1=len(VehicleFiles_test)


for i in range (1,len1):
 
 
 image_neg= cv2.imread(VehicleFiles_test[i])
 cv2.imshow(VehicleFiles_test[i],image_neg)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
 test_image = image.load_img(VehicleFiles_test[i],target_size = (64,64))
 test_image= image.img_to_array(test_image)
 test_image = np.expand_dims(test_image,axis=0) #since predict only works on
 result =model.predict(test_image)
 print('test')
 res =result[0][0]
 if (result[0][0] >  0.5):
    true_positive=true_positive+1
    print(' ok')
 else:
   false_negative=false_negative+1
   print('not ok')
   
   #############################################################
false_positive=0
true_negative=0

VehicleFiles_test = glob.glob('*.jpg')
len1=len(VehicleFiles_test)


for i in range (1,len1):
 
 
 image_neg= cv2.imread(VehicleFiles_test[i])
 cv2.imshow(VehicleFiles_test[i],image_neg)
 cv2.waitKey(0)
 cv2.destroyAllWindows()
 test_image = image.load_img(VehicleFiles_test[i],target_size = (64,64))
 test_image= image.img_to_array(test_image)
 test_image = np.expand_dims(test_image,axis=0) #since predict only works on
 result =model.predict(test_image)
 print('test')
 res =result[0][0]
 if (result[0][0] >  0.5):
    false_positive=false_positive+1
    print('not ok')
 else:
   true_negative=true_negative+1
   print('ok')
   
'''





 
















