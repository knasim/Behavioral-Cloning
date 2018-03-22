import csv

from keras.models import load_model

"""
author: Khurrum  Nasim
date:  03.18.18
description:  Retraining the model on newly collected data only for recovery action mode 
"""


path = "/home/ubuntu/retrain/set4/"
csv_binary = "driving_log.csv"
lines = []  # data dictionary
model_json = 'model.json'
model_weights = '/home/ubuntu/sandtests/model.h5'
model_retrain = 'model-retrained.h5'

#read in the csv file
with open(path + csv_binary) as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

import cv2
import numpy as np
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    correction = 0.2

    while True:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batches = samples[offset:offset+batch_size]

            images = []
            measurements = []

            for b in batches:
                if abs(float(b[3])) < 0.15:
                    continue
                center_image_source = path + 'IMG/'+b[0].split('/')[-1]
                center_image = cv2.imread(center_image_source)
                center_measurement = float(b[3])
                images.append(center_image)
                measurements.append(center_measurement)

                left_image_source = path + 'IMG/'+b[1].split('/')[-1]
                left_image = cv2.imread(left_image_source)
                left_measurement = center_measurement + correction
                images.append(left_image)
                measurements.append(left_measurement)

                right_image_source = path + 'IMG/'+b[2].split('/')[-1]
                right_image = cv2.imread(right_image_source)
                right_measurement = center_measurement - correction
                images.append(right_image)
                measurements.append(right_measurement)


            #trim image
            augmented_images, augmented_measurements = [], []
            for image, angle in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(angle)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(angle*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train,y_train)



model = load_model(model_weights)  # load the existing model
train_generator = generator(train_samples, batch_size=32) # train on newly collected data from recovery action mode
validation_generator = generator(validation_samples, batch_size=32)

model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
    validation_data=validation_generator, nb_val_samples=len(validation_samples),
    nb_epoch=10)

print(model.summary())
model.save(model_retrain)