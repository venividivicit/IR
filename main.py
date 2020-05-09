import tensorflow as tf
import os
import cv2
import random
import numpy as np 
import pickle


data_dir = "/Users/veni_vedi_veci/Projects/image-recognition/data"
categories = ["wolves", "sheep"]

for category in categories:
    path = os.path.join(data_dir, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)


IMG_SIZE = 50
new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))


#creating a training data set from images
training_data = []
def create_training_data():
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category) #0 is wolf, 1 is sheep
        for img in os.listdir(path): 
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) # resize image because some landscape and some are portrait
                training_data.append([new_img_array, class_num])
            except Exception as e:
                print(e)
                pass
create_training_data()


random.shuffle(training_data)


X = [] #features
y = [] #labels

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)



pickle_out = open("/Users/veni_vedi_veci/Projects/image-recognition/X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("/Users/veni_vedi_veci/Projects/image-recognition/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()