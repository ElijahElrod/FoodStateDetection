import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model
from xml.etree import ElementTree as ET
from pathlib import Path
import numpy as np
import random
import sys
import cv2

def filelist(root, file_type):
    """Returns a fully-qualified list of filenames under root directory"""
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(file_type)]

def create_imgs_labels_from_dir(path):
  
    test_files = filelist(path, '.jpg')
    test_img_labels = []
    print("Processing Test Set")
    for ind, i in enumerate(test_files):
        path = i.split("\\")
        label = path[1]
        img = cv2.imread(i)
        img_copy = img.copy()
        ROI = cv2.resize(img_copy, (128, 128), 1)
        grayscaled_ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        img_data = np.asarray(grayscaled_ROI).astype('float')

        test_img_labels.append([img_data, label])
    

   
    return test_img_labels

def main():
    test_set = create_imgs_labels_from_dir('./data/dataset1/valid')
    adam_model = load_model('./checkpoint/adam/')
   
    X_test = []
    y_test = []
    for features, labels in test_set:
        X_test.append(features)
        y_test.append(labels)

    X_test = np.array(X_test).reshape(-1, 128, 128, 1)
    X_test /= 255.0
    y_test = pd.get_dummies(y_test)

    adam_results = adam_model.evaluate(X_test, y_test, verbose=0)
    print("Loss: " + str(adam_results[0]) + ", Accuracy: " + str(adam_results[1]))

if __name__ == '__main__':
    main()