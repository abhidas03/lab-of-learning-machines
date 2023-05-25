#Done using tutorial at: https://www.datacamp.com/tutorial/convolutional-neural-networks-python

#Import all images 
from keras.datasets import fashion_mnist
(train_X, train_Y), (test_X, test_Y) = fashion_mnist.load_data()

#Analyze what the images in the dataset look like
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#plt.show()
print("Training data shape: ", train_X.shape, train_Y.shape)
print("Testing data shape: ", test_X.shape, test_Y.shape)

#Find the unique numbers from the training labels
classes = np.unique(train_Y)
nClasses = len(classes)
print("Total number of outputs: ", nClasses)
print("Output classes: ", classes)

#What are the images in the dataset? 
plt.figure(figsize=[5,5])
#Training data
#Subplot lets you create side-by-side plots:
#Figure has one row, two columns, this plot is first/second plot
plt.subplot(121)
#Displays data as an image 
plt.imshow(train_X[0,:,:], cmap="gray")
plt.title("Ground Truth: {}".format(train_Y[0]))
#Testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap="gray")
plt.title("Ground Truth: {}".format(test_Y[0]))
#plt.show()

#Resizing images for preprocessing
train_X = train_X.reshape(-1, 28, 28, 1)
test_X = test_X.reshape(-1, 28, 28, 1)
train_X = train_X.astype("float32")
test_X = test_X.astype("float32")
train_X = train_X/255
test_X = test_X/255

#Encode class types into vectors with 0s for everything except actual class
#Change labels from categorical to one-hot encoding
train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
print("Original label: ", train_Y[0])
print("After conversion: ", train_Y_one_hot[0])

#Partition training data into 80% training and 20% validation
from sklearn.model_selection import train_test_split
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size = 0.2, random_state=13)
print(train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)

#Import to train models
import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, LeakyReLU
)
batch_size = 64
epochs = 20
num_classes = 10

#Adding layers to the model
fashion_model = Sequential()
#32 total filters, 3x3 kernel size, zero-padding with dropout
fashion_model.add(Conv2D(32, (3,3), activation="linear", input_shape=(28, 28, 1), padding="same"))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2,2), padding="same"))
fashion_model.add(Conv2D(64, (3,3), activation="linear", padding="same"))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2),padding="same"))
fashion_model.add(Conv2D(128, (3,3), activation="linear", padding="same"))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
#Flatten transforms image into a single vector 
fashion_model.add(Flatten())
fashion_model.add(Dense(128, activation="linear"))
fashion_model.add(LeakyReLU(alpha=0.1))
#Activation function with 10 units needed for 10-class classification
fashion_model.add(Dense(num_classes, activation="softmax"))

fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.legacy.Adam(), metrics=["accuracy"])
fashion_model.summary()

#Training the model for 20 epochs
fashion_train = fashion_model.fit(train_X, train_label, batch_size = batch_size, epochs=epochs, verbose = 1, validation_data = (valid_X, valid_label))

#Model evaluation on the test set
test_eval = fashion_model.evaluate(test_X, test_Y_one_hot, verbose=0)
print("Test loss: ", test_eval[0])
print("Test accuracy: ", test_eval[1])

#Loss and Accuracy Plots
accuracy = fashion_train.history["acc"]
val_accuracy = fashion_train.history["val_acc"]
loss = fashion_train.history["loss"]
val_loss = fashion_train.history["val_loss"]
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, "bo", label = "Training Accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "b", label="Validation Loss")
plt.legend()
plt.show() 
fashion_model.save("fashion_model.h5py")

#Adding dropout
batch_size = 64
epochs = 20
num_classes = 10
fashion_model = Sequential()
fashion_model.add(Conv2D(32, kernel_size = (3,3), activation="linear", padding="same", input_shape=(28,28,1)))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D((2,2), padding="same"))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(64,(3,3), activation="linear", padding="same"))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
fashion_model.add(Dropout(0.25))
fashion_model.add(Conv2D(128, (3,3), activation="linear", padding="same"))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(MaxPooling2D(pool_size=(2,2), padding="same"))
fashion_model.add(Dropout(0.4))
fashion_model.add(Flatten())
fashion_model.add(Dense(128,activation="linear"))
fashion_model.add(LeakyReLU(alpha=0.1))
fashion_model.add(Dropout(0.3))
fashion_model.add(Dense(num_classes,activation="softmax"))
fashion_model.summary()
fashion_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=["accuracy"])
fashion_train_dropout = fashion_model.fit(train_X, train_label, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data=(valid_X, valid_label))
fashion_model.save("fashion_model_dropout.h5py")