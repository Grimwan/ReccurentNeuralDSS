import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#Imagesfiles map is needed for imageloader to run.
import random
import pickle 
class ImageLoader(object):
    def imageReader(dataDir, trainingDir, resultDir, resize):
        training_data = []
        for category in trainingDir:
            path = os.path.join(dataDir, category) # path to img/training or pixel-level-gt/training dir
            secondPath = os.path.join(dataDir,resultDir[trainingDir.index(category)])
            #class_num = CATEGORIES.index(category)
            newImg =  os.listdir(secondPath)
            i = 0;
            for img in os.listdir(path):
                try:
                    imgTraining_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
                    imgResult_array = cv2.imread(os.path.join(secondPath,newImg[i]),cv2.IMREAD_COLOR)
                    if resize[0] != 0:
                        newTraining_array = cv2.resize(imgTraining_array, (resize[1],resize[2]))
                        newResult_array = cv2.resize(imgResult_array, (resize[1],resize[2]))
                        training_data.append([newTraining_array,newResult_array])
                    else:
                        training_data.append([imgTraining_array,imgResult_array])
                    i= i+1
                except Exception as e:
                    pass
        return training_data

    def saveImages(dataDir, trainingDir, resultDir, resize,shuffleData,pickleOutX,pickleOutY):
        training_data = ImageLoader.imageReader(dataDir, trainingDir, resultDir, resize)
        if(bool(shuffleData)):
            random.shuffle(training_data)
        x = [] #trainingdata
        y = [] #labels
        for features, label in training_data:
            x.append(features)
            y.append(label)
        if(bool(resize[0])):
            x = np.array(x).reshape(-1,resize[1],resize[2],3)
            y = np.array(y).reshape(-1,resize[1],resize[2],3)
        else:
            x = np.array(x)
            y = np.array(y)
        pickle_out = open(pickleOutX,"wb")
        pickle.dump(x,pickle_out)
        pickle_out.close()

        pickle_out = open(pickleOutY,"wb")
        pickle.dump(y,pickle_out)
        pickle_out.close()
        print("Finished saving images to pickle x for "+pickleOutX+"  y for "+pickleOutY)

    def loadSavedImage(inX,inY):
        pickle_in = open(inX,"rb")
        x_train = pickle.load(pickle_in)
        pickle_in = open(inY,"rb")
        y_train = pickle.load(pickle_in)
        return [x_train,y_train]