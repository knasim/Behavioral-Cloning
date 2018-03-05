import csv

path = "/home/ubuntu/data_set_5/"
csv_binary = "driving_log.csv"
lines = []

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

    while True: # loop forever
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

# train using a generator
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

ch, row, col = 3, 80, 320  # trimmed format

from keras.models import Sequential
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D

#nvdia network
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))
model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dropout(0.5))
model.add(Dense(1))

# Adam optimizer
model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
    validation_data=validation_generator, nb_val_samples=len(validation_samples),
    nb_epoch=5)

print(model.summary())
model.save('model-github.h5')