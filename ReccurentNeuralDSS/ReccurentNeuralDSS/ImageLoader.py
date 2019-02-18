import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
#Imagesfiles map is needed for imageloader to run.
import random
import pickle 
class ImageLoader(object):
    def imageReader(dataDir, trainingDir, resultDir, resize,Crop):
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
                    max_y,max_x = imgTraining_array.shape[:2]
            #        print(max_x)
            #        print(max_y)
                    if resize[0] != 0:
                        imgTraining_array = cv2.resize(imgTraining_array, (resize[1],resize[2]))
                        imgResult_array = cv2.resize(imgResult_array, (resize[1],resize[2]))
                    if Crop[0] != 0:
                        border_y = 0
                        if max_y % Crop[2] != 0:
                            border_y = (Crop[2] - (max_y % Crop[2]) + 1) // 2
                            imgTraining_array = cv2.copyMakeBorder(imgTraining_array,border_y,border_y,0,0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                            imgResult_array = cv2.copyMakeBorder(imgResult_array,border_y,border_y,0,0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        border_x = 0
                        if max_x % Crop[1] != 0:
                            border_x = (Crop[1] - (max_x % Crop[1]) + 1) // 2
                            imgTraining_array = cv2.copyMakeBorder(imgTraining_array,0,0,border_x,border_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                            imgResult_array = cv2.copyMakeBorder(imgResult_array,0,0,border_x,border_x, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        curr_y = 0                   
                        parts = []
                        while(curr_y + Crop[2])<= max_y:
                            curr_x = 0
                            while(curr_x + Crop[1]) <= max_x:
                                training_data.append([imgTraining_array[curr_y:curr_y+Crop[2],curr_x:curr_x + Crop[1]],imgResult_array[curr_y:curr_y+Crop[2],curr_x:curr_x + Crop[1]]])
                                curr_x += Crop[1]
                            curr_y += Crop[2]
                    else:
                        training_data.append([imgTraining_array,imgResult_array])
                    i= i+1
                except Exception as e:
                    pass
        return training_data

    def saveImages(dataDir, trainingDir, resultDir, resize,Crop,shuffleData,pickleOutX,pickleOutY,WheretosavePickleData):
        training_data = ImageLoader.imageReader(dataDir, trainingDir, resultDir, resize,Crop)
       # if(bool(shuffleData)):
        #    random.shuffle(training_data)
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

        im2 = x.copy()
        im2[:,:, :, 0] = x[:, :, :, 2]
        im2[:,:, :, 2] = x[:, :, :, 0]
        x = im2
        ANotherImg = y.copy()
        ANotherImg[:,:, :, 0] = y[:,:, :, 2]
        ANotherImg[:,:, :, 2] = y[:,:, :, 0]
        y = ANotherImg
        pickle_out = open(WheretosavePickleData+pickleOutX,"wb")
        pickle.dump(x,pickle_out)
        pickle_out.close()

        pickle_out = open(WheretosavePickleData+pickleOutY,"wb")
        pickle.dump(y,pickle_out)
        pickle_out.close()
        print("Finished saving images to pickle x for "+pickleOutX+"  y for "+pickleOutY)

    def loadSavedImage(WhereToload,inX,inY):
        pickle_in = open(WhereToload+inX,"rb")
        x_train = pickle.load(pickle_in)
        pickle_in = open(WhereToload+inY,"rb")
        y_train = pickle.load(pickle_in)
        return [x_train,y_train]

    def combine_imgs(imgs, max_y: int, max_x: int) -> np.array:
        img = np.zeros((max_y,max_x,3),np.int)#Img = np.zeros((max_y, max_x), np.float)
        #np.array(x).reshape(-1,resize[1],resize[2],3)
        size_y, size_x,nope = imgs[0].shape
        curr_y = 0
        i = 0
        # TODO: rewrite with generators.
        while (curr_y + size_y) <= max_y:
            curr_x = 0
            while (curr_x + size_x) <= max_x:
                try:
                    img[curr_y:curr_y + size_y, curr_x:curr_x + size_x] = imgs[i]
                except:
                    i -= 1
                i += 1
                curr_x += size_x
            curr_y += size_y
        return img
