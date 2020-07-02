"""This script, is meant to be used to make a data set inside a classes folder that has to be in the same location
as this file. To call this script, you must specify the positional arguments. the first one , is the number of samples
you want to take. And the second one, the name of the samples. This code, will save the images with the samples names
and a number starting form zero to the first positional argument minus 1."""

# Import libraries.
import cv2 as cv  # Computer vision library.
from picamera.array import PiRGBArray  # PiCamera libraries.
from picamera import PiCamera
from time import sleep  # Time library.
import numpy as np  # Array library.
import argparse

# Positional arguments setup.
parser = argparse.ArgumentParser(description='Data set values to use.')
parser.add_argument('samples', action='store', help='Number of pictures to take for the data set')
parser.add_argument('name', action='store', help='Enter the name of the class for the data set')
arguments = parser.parse_args()

# Camera setup.
camera = PiCamera()
camera.resolution = (640, 480)  # Setup camera resolution.
camera.framerate = 90  # FPS camera setup.
rawCapture = PiRGBArray(camera, size=(640, 480))
sleep(0.1)  # Camera warm up time.

# Variable setup.
samples = int(arguments.samples)
name = arguments.name
index = 'classes/' + name  # Path to save images.
counter = 0  # Number of image taken.
flag = 0  # Flag to take images.

# Color filter values.
lower_yellow = np.array([22, 92, 0])
upper_yellow = np.array([45, 255, 255])


# Image capture.
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array  # Taking a frame.
    cv.imshow('Capture', image)

    # Image filtering.
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    processed_image = cv.medianBlur(mask, 9)
    processed_image = cv.blur(processed_image, (5, 5))
    cv.imshow('Processed image', processed_image)

    # Image saving flag.
    path = index + str(counter) + '.jpg'
    if cv.waitKey(1) & 0xFF == ord('q'):
        flag = 1
        print('Writing...')
        sleep(10)  # Idle time to setup.
    else:
        pass

    # Image saving.
    if flag == 1:
        cv.imwrite(path, processed_image)
        counter += 1
        print('{} images saved.'.format(counter))

        if counter == samples:
            break

    # Update the frame.
    rawCapture.truncate(0)

# Destroy the window when finished.
cv.destroyAllWindows()
