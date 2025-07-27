import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from config import img_height, img_width

def pathGeneration(trainPath):
    xTrain, yTrain = {}, {}
    for idx, name in enumerate(os.listdir(trainPath)):
        xTrain[idx] = os.path.join(trainPath, name, 'images', f'{name}.png')
        mask_dir = os.path.join(trainPath, name, 'masks')
        for mask_file in os.listdir(mask_dir):
            yTrain.setdefault(idx, []).append(os.path.join(mask_dir, mask_file))
    return xTrain, yTrain

def preProcess(img_path):
    img = imread(img_path)[..., :3]
    return resize(img, (img_height, img_width))

def maskFormation(mask_paths):
    mask = np.zeros((img_height, img_width, 1), dtype=np.float32)
    for p in mask_paths:
        m = resize(imread(p), (img_height, img_width))
        mask = np.maximum(mask, np.expand_dims(m, -1))
    return mask

def dataGen(xList, yList, batchSize):
    while True:
        idxs = np.random.choice(len(xList), batchSize)
        X, Y = [], []
        for i in idxs:
            X.append(preProcess(xList[i]))
            Y.append(maskFormation(yList[i]))
        yield np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def split_data(xTrain, yTrain, test_size=0.2):
    ids = list(xTrain.keys())
    train_ids, val_ids = train_test_split(ids, test_size=test_size, random_state=42)
    xTrain_split = [xTrain[i] for i in train_ids]
    yTrain_split = [yTrain[i] for i in train_ids]
    xVal_split   = [xTrain[i] for i in val_ids]
    yVal_split   = [yTrain[i] for i in val_ids]
    return xTrain_split, yTrain_split, xVal_split, yVal_split
