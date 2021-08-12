
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical, plot_model
from keras import Input
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
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
 
        img_label_list.append([img_data, row['state']])
        img_label_list.append([flip_1, row['state']])
        img_label_list.append([rotate_1, row['state']])
        img_label_list.append([rotate_2, row['state']])
        img_label_list.append([rotate_3, row['state']])   

    return img_label_list


def create_imgs_labels_from_dir():
    train_files = filelist('./data/dataset1/train', '.jpg')
    test_files = filelist('./data/dataset1/valid', '.jpg')
    train_img_labels = []
    print("Processing Training Files")
    for ind, i in enumerate(train_files):
        path = i.split("\\")
        label = path[1]
        img = cv2.imread(i)
        img_copy = img.copy()
        ROI = cv2.resize(img_copy, (128, 128), 1)
        grayscaled_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        img_data = np.asarray(grayscaled_ROI).astype('float')

        flip_1 = np.fliplr(img_data)
        rotate_1 = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
        rotate_2 = cv2.rotate(img_data, cv2.ROTATE_180)
        rotate_3 = cv2.rotate(img_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
        train_img_labels.append([img_data, label])
        train_img_labels.append([flip_1, label])
        train_img_labels.append([rotate_1, label])
        train_img_labels.append([rotate_2, label])
        train_img_labels.append([rotate_3,label])  

    test_img_labels = []
    print("Processing Test Files")
    for ind, i in enumerate(test_files):
        path = i.split("\\")
        label = path[1]
        img = cv2.imread(i)
        img_copy = img.copy()
        ROI = cv2.resize(img_copy, (128, 128), 1)
        grayscaled_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        img_data = np.asarray(grayscaled_ROI).astype('float')

        flip_1 = np.fliplr(img_data)
        rotate_1 = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
        rotate_2 = cv2.rotate(img_data, cv2.ROTATE_180)
        rotate_3 = cv2.rotate(img_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
 
        test_img_labels.append([img_data, label])
        # test_img_labels.append([flip_1, label])
        # test_img_labels.append([rotate_1, label])
        # test_img_labels.append([rotate_2, label])
        # test_img_labels.append([rotate_3,label])

    return train_img_labels, test_img_labels



def make_conv_model(input_shape, opt):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=(3,3),
                     activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
  
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(.33))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(.33))
  

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Dropout(.33))
    model.add(Dense(11, activation='softmax'))
    model.compile(optimizer=opt, loss="categorical_crossentropy",
                  metrics=['accuracy'])
    model.summary()
    return model


def main():

    IMG_SIZE = 128
    BATCH_SIZE = 64
    EPOCHS = 50
    MDL_CHK_PT_PATH = './checkpoint'
    sgd_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{MDL_CHK_PT_PATH}/sgd',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch')

    adam_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=f'{MDL_CHK_PT_PATH}/adam',
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_freq='epoch')

    input_shape = (IMG_SIZE, IMG_SIZE, 1)

    anno_path = Path('./data/Annotations')
    images_path = Path('./data/Images')
    train_path = Path('./data/train')
    test_path = Path('./data/test')

    train_set, test_set = create_imgs_labels_from_dir()
    # data_df = generate_data_df(anno_path, images_path)
    # imgs_labels = create_image_label_df(data_df)
 
    '''
    Training set distribution in case needed to up or downsample a category
    '''
    # labelDist = {}
    # for x,y  in train_set:
    #     if labelDist.get(y) is None:
    #         labelDist[y] = 1
    #     else:
    #         labelDist[y] = labelDist.get(y)+1

    # for i in labelDist:
    #     labelDist[i] = labelDist[i]/len(train_set)
    # print(labelDist)

    random.shuffle(train_set)
    random.shuffle(test_set)
    

    X_train = []
    y_train = []
    for features, labels in train_set:
        X_train.append(features)
        y_train.append(labels)

    X_test = []
    y_test = []
    for features, labels in test_set:
        X_test.append(features)
        y_test.append(labels)
  
 
    X_train = np.array(X_train).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_train /= 255.0
    X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    X_test /= 255.0

    y_train = pd.get_dummies(y_train)   
    y_test = pd.get_dummies(y_test)

   

    adam_tensorboard_callback = TensorBoard(log_dir="logs/adam/{}".format("ADAM CNN"))
    sgd_tensorboard_callback = TensorBoard(log_dir="logs/sgd/{}".format("SGD CNN"))

    adam_model = make_conv_model(input_shape=input_shape, opt="adam")
    adam_model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[adam_tensorboard_callback, adam_model_checkpoint_callback])

    sgd_model = make_conv_model(input_shape=input_shape, opt="sgd")
    sgd_model.fit(X_train, y_train,  validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[sgd_tensorboard_callback, sgd_model_checkpoint_callback])

    

  

if __name__ == "__main__":
    main()
