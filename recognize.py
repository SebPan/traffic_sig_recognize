"""This script, use the model.json and model.h5 files, that contains the neural network, to make decisions based on what
 the robot is seeing."""

# Import the libraries.
from time import sleep
import cv2 as cv
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
from tensorflow.keras.models import model_from_json
import RPi.GPIO as gpio
import argparse

# Argument setup.
parser = argparse.ArgumentParser(description='Acceptation percentage of the model.')
parser.add_argument('acceptation', action='store', help='Enter the acceptation percentage between 0 and 1 for the model')
argument = parser.parse_args()

# Camera setup.
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 90
rawCapture = PiRGBArray(camera, size=(640, 480))
sleep(0.1)

# Variable declaration.
image_dim = 50
image_input = np.zeros((1, image_dim * image_dim))
acceptation = float(argument.acceptation)
acceptation_range = acceptation - 0.05
turn_time = 0.7

# Filter values
lower_yellow = np.array([22, 93, 0])
upper_yellow = np.array([45, 255, 255])

# DC motor gpio pins.
left_forward = 3
left_backward = 2
right_forward = 17
right_backward = 27
left_enable = 23
right_enable = 24

text = ''

# GPIO setup.
gpio.setmode(gpio.BCM)
gpio.setup(left_forward, gpio.OUT)
gpio.setup(left_backward, gpio.OUT)
gpio.setup(right_forward, gpio.OUT)
gpio.setup(right_backward, gpio.OUT)
gpio.setup(left_enable, gpio.OUT)
gpio.setup(right_enable, gpio.OUT)
left_PWM = gpio.PWM(left_enable, 1000)
right_PWM = gpio.PWM(right_enable, 1000)
left_PWM.start(75)
right_PWM.start(55)


# Forward function.
def forward():
    gpio.output(left_forward, gpio.HIGH)
    gpio.output(right_forward, gpio.HIGH)
    gpio.output(left_backward, gpio.LOW)
    gpio.output(right_backward, gpio.LOW)


# Right function.
def right():
    gpio.output(left_forward, gpio.HIGH)
    gpio.output(right_forward, gpio.LOW)
    gpio.output(left_backward, gpio.LOW)
    gpio.output(right_backward, gpio.LOW)


# Left function.
def left():
    gpio.output(left_forward, gpio.LOW)
    gpio.output(right_forward, gpio.HIGH)
    gpio.output(left_backward, gpio.LOW)
    gpio.output(right_backward, gpio.LOW)


# Stop function.
def stop():
    gpio.output(left_forward, gpio.LOW)
    gpio.output(right_forward, gpio.LOW)
    gpio.output(left_backward, gpio.LOW)
    gpio.output(right_backward, gpio.LOW)


# Model load.
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
loaded_model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
print('Model loaded.')

# Video start.
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    image = frame.array

    # Image setup.
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv, lower_yellow, upper_yellow)
    mask = cv.medianBlur(mask, 9)
    mask = cv.blur(mask, (5, 5))
    resized_gray_image = cv.resize(mask, (image_dim, image_dim))
    image_input[0, :] = resized_gray_image.flatten() / 255.0

    # Image_prediction.
    image_prediction = loaded_model.predict(x=image_input, batch_size=1)
    print(image_prediction)

    # Image labeling and decision making.
    if image_prediction[0, 0] > acceptation:
        text = 'Stop.'
        stop()
    if image_prediction[0, 1] > acceptation:
        text = 'Right.'
        right()
        sleep(turn_time + 0.3)
        stop()
    if image_prediction[0, 2] > acceptation:
        text = 'Left.'
        left()
        sleep(turn_time)
        stop()
    if image_prediction[0, 3] > acceptation:
        text = 'No traffic sign.'
        forward()
    if image_prediction[0, 0] < acceptation_range and image_prediction[0, 1] < acceptation_range and image_prediction[0, 2] < acceptation_range and image_prediction[0, 3] < acceptation_range:
        text = 'No traffic sign.'
        forward()

    # Image show.
    text_image = cv.putText(image, 'Prediction: ' + text, (30, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow('Prediction', image)
    cv.imshow('Mask', mask)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    rawCapture.truncate(0)

# Clean windows and gpio.
cv.destroyAllWindows()
gpio.cleanup()
