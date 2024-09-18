"""REMINDER:
    - Change the constants to improve model performance
    - Change the dataset path to the path of the dataset folder on your computer
    - Change the test path to the path of the test folder on your computer
    - Dataset folder and test folder must include label names as folder names
    - If the model has been trained before, delete the model file in the current folder and restart
    - Usage: {python model.py (test|main)} in the terminal
"""

# Install dependencies if not already installed (pip3 install -r requirements.txt)
import csv
import os
import cv2
import numpy as np
import keras.losses
import keras.optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import tensorflow as tf

# Number of classes in dataset (must match the classes amount in the dataset) and image dimensions
NUM_CLASSES = 3
IMG_WIDTH = 120
IMG_HEIGHT = 120

# Change these constants to improve model performance
BATCH_SIZE = 16
NUM_EPOCHS = 15

# Dataset path and test path
DATASET_PATH = 'dataset'
TEST_PATH = 'test'

# Classes in dataset (must match folder names in dataset path and test path)
CLASSES = ['barcode', 'qr', 'else']

# Model path and name
MODEL_PATH = 'models/'
MODEL_NAME = 'model2'

def load_dataset(path):
    """Loads dataset from path and returns data and labels
    path: path to dataset folder
    
    Returns: data, labels"""
    # Check if path is valid
    if path is None:
        print('Invalid path')
        exit()

    # Initialize data and labels
    data = []
    labels = []

    # Loop through each folder in the dataset
    for folder_name in os.listdir(path):
        folder_path = os.path.join(path,folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path,file_name)

            print("Loading file: " + file_path + " ...")

            # Check if file path is valid and is an image
            if file_path is None and file_path != '.jpg':
                print('Invalid file path')
                continue
            img = cv2.imread(file_path)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

            # Add image to data
            data.append(img)

            # Assign label to image
            label = folder_name
            for i, class_name in enumerate(CLASSES):
                if label == class_name:
                    print('Label: ' + str(i) + ' (' + class_name + ')')
                    label = i
                    break
            labels.append(label)

    return np.array(data), np.array(labels)

def split_dataset(x_data, y_label):
    """Splits dataset into training, validation, and testing sets
    x: data
    y: labels
    
    Returns: training, validation, and testing sets of data and labels"""
    # Shuffle data
    num_samples = len(x_data)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    split1 = int(num_samples * 0.7)
    split2 = int(num_samples * 0.85)

    # Split data into training, validation, and testing sets
    x_train, y_train = x_data[indices[:split1]], y_label[indices[:split1]]
    x_val, y_val = x_data[indices[split1:split2]], y_label[indices[split1:split2]]
    x_test, y_test = x_data[indices[split2:]], y_label[indices[split2:]]

    return x_train, y_train, x_val, y_val, x_test, y_test

def make_model():
    """Builds and trains CNN model and saves model to file
    
    Returns: None"""
    # Build CNN model
    model = Sequential()
    model.add(Conv2D(32,
                    kernel_size=(3, 3),
                    activation='relu',
                    input_shape=(IMG_WIDTH,IMG_HEIGHT,
                    3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,
                    kernel_size=(3, 3),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128,
                    activation='relu'))
    model.add(Dense(units=NUM_CLASSES,
                    activation='softmax'))

    # Compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])

    # Load and preprocess data
    x_data, y_label = load_dataset('dataset')

    # Add more data by flipping images randomly (horizontal and vertical)
    x_data = np.concatenate((x_data, np.flip(x_data, 1)), axis=0)
    y_label = np.concatenate((y_label, y_label), axis=0)
    x_data = np.concatenate((x_data, np.flip(x_data, 2)), axis=0)
    y_label = np.concatenate((y_label, y_label), axis=0)

    # Split data into training, validation, and testing sets
    x_train, y_train, x_val, y_val, x_test, y_test = split_dataset(x_data, y_label)
    x_train = x_train.astype('float32') / 255
    x_val = x_val.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices (one-hot encoding)
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(y_val, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Train model
    model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=NUM_EPOCHS,
            verbose=1,
            validation_data=(x_val, y_val))

    # Evaluate model
    score = model.evaluate(x_test, y_test, verbose=0)
    loss = score[0]
    accuracy = score[1]
    error = 1 - score[1]

    # Output results to file
    with open(MODEL_PATH + '/' + MODEL_NAME + '/results.txt', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Loss', 'Accuracy', 'Error'])
        writer.writerow([loss, accuracy, error])

        print("Path: " + MODEL_PATH + '/' + MODEL_NAME + '/results.txt')

    # Save model
    model.save(MODEL_PATH + '/' + MODEL_NAME) # savedmodel

def load_model(model_path):
    """Loads model from file
    model_path: path to model file
    
    Returns: model"""
    if model_path is None:
        print('Invalid model path')
        return None

    model = tf.keras.models.load_model(model_path)
    return model

def predict(model, img):
    """Predicts class of image
    model: model to use for prediction
    img: image to predict
    
    Returns: array of predictions"""
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

def main():
    """Main function"""
    # Create model directory if it doesn't exist
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)
        os.makedirs(MODEL_PATH + '/' + MODEL_NAME)

    # If model name already exists, ask user if they want to overwrite
    if os.path.exists(MODEL_PATH + '/' + MODEL_NAME + '.h5'):
        print('Model already exists. Overwrite? (y/n)')
        overwrite = input()
        if overwrite == 'y':
            make_model()
        else:
            print('Model not overwritten')
    else:
        # Build and train model if it doesn't exist
        make_model()

if __name__ == '__main__':
    main()
