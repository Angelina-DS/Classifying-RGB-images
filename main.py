import matplotlib.pyplot as plt 
import matplotlib
import tkinter
matplotlib.use('TkAgg')
import random
import numpy as np 
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential #The model type we'll use
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras import regularizers

##Loading the data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
#or (x_train, y_train), (x_test, y_test) = cifar10.load_data()

##Displaying some examples and describing the dataset
for i in range(9):
    num = random.randint(0, len(train_images))
    plt.imshow(train_images[num], cmap='gray', interpolation='none')
    plt.title("Class {}".format(train_labels[num]))
    #plt.show() putting an hashtag to avoid showing 10 images each time the code is running

plt.tight_layout() #adjust the padding between and around the subplots

print('')
print('The CIFAR database contains {} images for training and {} images for testing/validation'.format(len(train_images), len(test_images)))
print('train_images/test_images are composed of four dimensions.')
print('For example, the ones of train_images are {} and refer respectively to:'.format(train_images.shape))
print('- the number of images in the database')
print('- the 2D size of the images (32*32)')
print('- the number of channels (for example the images are RGB and contain 3 channels)')
print('')

im = train_images[0]
print('The size of the image is {} pixels'.format(im.shape))
print('The minimum pixel value is {} and the maxium  {}'.format(np.min(im), np.max(im)))

##Formating the data
"""As we said previously and in contrary to the MNIST dataset, from the CIFAR database, 
the images are composed of 3 channels (RGB). Thus each channel needs to be normalized separately
in order to work with normalized images."""

test_images_original = test_images #keeps the original validation images for later
#Calculations of the average and the standard deviation for each channel :
train_images = train_images.astype(float)/255
test_images = test_images.astype(float)/255

Mean_r = np.mean(train_images[:,:,:,0]) #The mean for the red channel
Mean_g = np.mean(train_images[:,:,:,1])
Mean_b = np.mean(train_images[:,:,:,2])

Std_r = np.std(train_images[:,:,:,0])
Std_g = np.std(train_images[:,:,:,1])
Std_b = np.std(train_images[:,:,:,2])

#Normalize data so that average is 0 and std 1 :
train_images[:,:,:,0] = (train_images[:,:,:,0] - Mean_r)/Std_r
train_images[:,:,:,1] = (train_images[:,:,:,1] - Mean_g)/Std_g
train_images[:,:,:,2] = (train_images[:,:,:,2] - Mean_b)/Std_b

test_images[:,:,:,0] = (test_images[:,:,:,0] - Mean_r)/Std_r
test_images[:,:,:,1] = (test_images[:,:,:,1] - Mean_g)/Std_g
test_images[:,:,:,2] = (test_images[:,:,:,2] - Mean_b)/Std_b

#Changing the labels from single digit to categorical or one-hot format :
#print(train_labels[0])
train_labels = np_utils.to_categorical(train_labels)
test_labels = np_utils.to_categorical(test_labels)
#print(train_labels[0])

##Creating and training a simple network for digit classification
"""We will now build a CNN able to read the CIFAR images as input and return a vector 
indicating the predicted class for each image."""

#Six convolution layers and one fully connected layer
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)), #64 different 3*3 kernels
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)), #Pool the max values over a 2*2 kernel

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(256, (3, 3), activation='relu', padding='same'),
    Conv2D(256, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),

    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10),
    Activation('softmax'),
])

#Compiling the model by defining the optimizer and the loss function 
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() # Returns a full description of the network

#Training the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=1, validation_data=(test_images,test_labels))

##Evaluate the model accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels) #Calculates the accuracy of the model using the test set
print('test_acc:', test_acc)

"""Checking the performance of the network on randomly selected images can be useful to improve
the network. We are going to test the network on the validation set and select images for which 
the network prediction is right and other for which the prediction is wrong."""

#The predicted_classes function outputs the highest probability class according to the trained classifier for each input example.
predicted_classes = model.predict(test_images) 
np.set_printoptions(suppress=True)
print('Output of the network for image #0:')
print(predicted_classes[0])

predicted_classes = np.argmax(predicted_classes, axis=1)
print('')
print('And we are only keeping the index of the highest value : {}'.format(predicted_classes[0]))

#Checking which items we got right / wrong
correct_indices = np.nonzero(predicted_classes == test_labels[:,0])[0]
incorrect_indices = np.nonzero(predicted_classes != test_labels[:,0])[0]
print('')
print('Finally the images are sorted :')
print('Number of properly indentified images : {}'.format(len(correct_indices)))
print('Number of wrongly indentified images : {}'.format(len(incorrect_indices)))

#Few examples of right predictions :
plt.figure()
for i, correct in enumerate(correct_indices[:3]):
    plt.imshow(test_images_original[correct], interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], test_labels[correct][0]))
    plt.show()

#Few examples of wrong predictions :
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:3]):
    plt.imshow(test_images_original[incorrect], interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], test_labels[incorrect][0]))
    plt.show()
    