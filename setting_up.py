"""This script, must be used to transform the images data set in the classes directory into a NumPy array and save it
into a comma-separated values file (.csv) called image_input.csv. This code has some user inputs, the first one,
specifies the quantity of samples to transform per class, the second input, asks to enter the quantity of classes to
be transformed, and finally you have to enter the prefix of all the classes (without the number of samples). """

# Import the libraries.
import cv2 as cv
import numpy as np

# Variables setup.
samples = int(input('Enter the number of samples per class: '))
quantity = int(input('Enter the quantity of classes: '))
names = [input('Enter the name of the class: ') for name in range(quantity)]
index = 0
counter = 0
image_dim = 50
labels = np.zeros((quantity * samples, 1))
image_input = np.zeros((quantity * samples, image_dim * image_dim))

# Path.
for i in names:
    for j in range(0, samples):
        path = 'classes/' + i + str(j) + '.jpg'
        print(path)

        # Image setting.
        image = cv.imread(path)
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        resized_gray_image = cv.resize(gray_image, (image_dim, image_dim))
        image_input[index:] = resized_gray_image.flatten()/255.0

        labels[index:] = counter
        index += 1

        cv.imshow('Image', resized_gray_image)
        cv.waitKey(10)

    counter += 1

np.savetxt('image_input.csv', image_input, delimiter=',')

