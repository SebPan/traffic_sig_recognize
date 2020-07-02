"""This Python code, is used to make the learning process of the neural network and save it into a javascript object
notation file as model.json and save the weights into a hierarchical data format file as model.h5. The scripts run
starting from the image_input.csv file created in the setting_up.py file. It has two user inputs. The first of them asks
to enter the quantity of classes that are going to be used in the learning process and the second one, waits for the
total number of samples to be analyzed."""

# Import libraries.
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Variable setup.
data = []
labels = []
counter = 0
quantity = int(input('Enter the number of classes: '))
samples = int(input('Enter the total number of samples: '))
data_cv = genfromtxt('image_input.csv', delimiter=',')
samples_per_class = int(samples / quantity)

# Data setup.
# Data from .csv to an NumPy array.
for i in range(0, samples):
    data.append(data_cv[i, :])
    labels.append(counter)
    if (i + 1) % samples_per_class == 0:
        counter += 1

data = np.array(data, dtype='float')
labels = np.array(labels)

# Train and Test images setup.
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.10)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# Training setup.
model = Sequential()
model.add(Dense(128, input_shape=(2500,), activation='relu'))  # Input layer.
model.add(Dense(64, activation='relu'))  # First hidden layer.
model.add(Dense(16, activation='sigmoid'))  # Second hidden layer.
model.add(Dense(len(lb.classes_), activation='softmax'))  # Output layer.

# Model training.
epochs = int(input('Enter the number of epochs to train: '))
model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), epochs=epochs)
print(model.evaluate(testX, testY))
print(model.predict(x=testX))

# Save the model.
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)  # Save the neural network into a Javascript object notation file.
model.save_weights('model.h5')  # Save the weights into a hierarchical data format file.
print('Model saved.')
