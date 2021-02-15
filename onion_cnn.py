
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from keras import Input
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from xml.etree import ElementTree as ET
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle as pkl
import random
import cv2

def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]


def generate_data_df(anno_path, images_path):
    print("Gathering Annotations")
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for ind, anno_path in enumerate(annotations):
        root = ET.parse(anno_path).getroot()
        anno = {}
        anno['image'] = ind
        anno['filename'] = Path(
            str(images_path) + '/' + root.find("./filename").text)
        anno['width'] = root.find("./size/width").text
        anno['height'] = root.find("./size/height").text

        # Find all object tags
        if (len(root.findall('./object')) != 0):
            for box in root.findall('./object'):
                details = {}
                details['xmin'] = float(box.find("./bndbox/xmin").text)
                details['ymin'] = float(box.find("./bndbox/ymin").text)
                details['xmax'] = float(box.find("./bndbox/xmax").text)
                details['ymax'] = float(box.find("./bndbox/ymax").text)
                details['state'] = box.find(
                    "./attributes/attribute/value").text
                details.update(anno)
                anno_list.append(details)
        else:  # Also need to classify the empty ones
            noLabelDetails = {}
            noLabelDetails['xmin'] = 0
            noLabelDetails['ymin'] = 0
            noLabelDetails['xmax'] = anno['width']
            noLabelDetails['ymax'] = anno['height']
            noLabelDetails['state'] = 'none'
            noLabelDetails.update(anno)
            anno_list.append(noLabelDetails)

    return pd.DataFrame(anno_list)


def create_image_label_df(df):
    img_label_list = []
    for ind, row in df.iterrows():

        file_path = row['filename']
        image = cv2.imread(file_path.__str__())
        img_copy = image.copy()

        x, y, w, h = float(row['xmin']), float(
            row['ymin']), float(row['xmax']), float(row['ymax'])

        bounded_img = image[int(y):int(h), int(x):int(w)]
        # Resize to constant img size 128x128
        ROI = cv2.resize(bounded_img, (128, 128), 1)
        grayscaled_ROI = cv2.cvtColor(
            ROI, cv2.COLOR_BGR2GRAY)  # Grayscale image

        new_path = f'./data/TrainImages/img_{ind}.png'
        cv2.imwrite(new_path, grayscaled_ROI)
        img_data = np.asarray(grayscaled_ROI).astype('float')

        # Perform Data augmentation
        flip_1 = np.fliplr(img_data)
        rotate_1 = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
        rotate_2 = cv2.rotate(img_data, cv2.ROTATE_180)
        rotate_3 = cv2.rotate(img_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        plt.imshow(img_data)
        plt.imshow(flip_1)
        img_label_list.append([img_data, row['state']])
        img_label_list.append([flip_1, row['state']])
        img_label_list.append([rotate_1, row['state']])
        img_label_list.append([rotate_2, row['state']])
        img_label_list.append([rotate_3, row['state']])
   
   
        

    return img_label_list




def make_conv_model(input_shape):
    model = Sequential(
      
    )

    model.add(Conv2D(128, kernel_size=3,
                     activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.5))

    model.add(Conv2D(64, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.5))

    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.5))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.summary()
    return model


def main():
    IMG_SIZE = 128

    BATCH_SIZE = 32
    EPOCHS = 20
    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    anno_path = Path('./data/Annotations')
    images_path = Path('./data/Images')
    train_path = Path('./data/TrainImages')

    class_dict = {'whole': 0, 'cut': 1, 'sliced': 2, 'chopped': 3,
                  'minced': 4, 'peeled': 5, 'other': 6, 'none': 7}

    data_df = generate_data_df(anno_path, images_path)
    imgs_labels = create_image_label_df(data_df)
    print(len(imgs_labels))

    random.shuffle(imgs_labels)

    X = []
    y = []
    for features, labels in imgs_labels:
        X.append(features)
        y.append(labels)
  
 
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X /= 255.0
    y_encoded = pd.get_dummies(y)   


    tensorboard = TensorBoard(log_dir="logs/{}".format("Object State CNN"))

    model = make_conv_model(input_shape=input_shape)
    model.fit(X, y_encoded, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.3, callbacks=[tensorboard])

    model.save('trained_model')
  

if __name__ == "__main__":
    main()
