import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle 
import utils.config as conf 

class ImageLoader():
    """Load dataset images, split them inte chunks and write them to pickle files."""
    
    def read_images_resize_crop(dataDir: str, trainingDir: str, resultDir: str, resize, crop):
        training_data = []
        
        for category in trainingDir:
            path = os.path.join(dataDir, category) # path to img/training or pixel-level-gt/training dir
            secondPath = os.path.join(dataDir, resultDir[trainingDir.index(category)])
            #class_num = CATEGORIES.index(category)
            newImg =  os.listdir(secondPath)
            i = 0;
            for img in os.listdir(path):
                try:
                    imgTraining_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    imgResult_array = cv2.imread(os.path.join(secondPath, newImg[i]), cv2.IMREAD_COLOR)
                    #first Resize
                    if resize[0] != 0:
                        imgTraining_array = cv2.resize(imgTraining_array, (resize[1], resize[2]))
                        imgResult_array = cv2.resize(imgResult_array, (resize[1], resize[2]))
                    if crop[0] != 0:
                        #First Try to add a boarder
                        max_y, max_x = imgTraining_array.shape[:2] # this value is taken out from the original picture and used later for setting the max values
                        imgTraining_array = ImageLoader.add_border(imgTraining_array, crop[1], 
                                                                  crop[2], [0,0,0], max_x, max_y)
                        imgResult_array = ImageLoader.add_border(imgResult_array, crop[1], 
                                                                crop[2], [0,0,0], max_x, max_y)
                        #Lets Split the images
                        imgTraining_array = ImageLoader.split_image(imgTraining_array, crop[1], crop[2], 
                                                                   max_x, max_y)
                        imgResult_array = ImageLoader.split_image(imgResult_array, crop[1], crop[2], 
                                                                 max_x, max_y)
                        training_data += list(zip(imgTraining_array, imgResult_array))
                    else:
                        training_data.append([imgTraining_array, imgResult_array])
                    i= i+1
                except Exception as e:
                    pass
        return training_data
       
    def image_reader(dataDir: str, trainingDir: str, resultDir: str):
        training_data = []
        
        for category in trainingDir:
            path = os.path.join(dataDir, category) # path to img/training or pixel-level-gt/training dir
            secondPath = os.path.join(dataDir, resultDir[trainingDir.index(category)])
            #class_num = CATEGORIES.index(category)
            newImg =  os.listdir(secondPath)
            i = 0;
            for img in os.listdir(path):
                try:
                    imgTraining_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                    imgResult_array = cv2.imread(os.path.join(secondPath, newImg[i]), cv2.IMREAD_COLOR)
                    #first Resize
                    training_data.append([imgTraining_array, imgResult_array])
                    i = i+1
                except Exception as e:
                    pass
        return training_data

    def load_and_save_images_to_pickle(dataDir: str, trainingDir: str, resultDir: str, resize, 
                                  crop, shuffleData, pickleOutX, pickleOutY, WheretosavePickleData):
        training_data = ImageLoader.read_images_resize_crop(dataDir, trainingDir, resultDir, resize,crop)
       
        # if(bool(shuffleData)):
        #    random.shuffle(training_data)
        x = [] #trainingdata
        y = [] #labels
        
        for features, label in training_data:
            x.append(features)
            y.append(label)
     
        if(resize[0] == 1 and crop[0] == 0):
            x = np.array(x).reshape(-1, resize[1], resize[2], 3)
            y = np.array(y).reshape(-1, resize[1], resize[2], 3)
        else:
            x = np.array(x)
            y = np.array(y)

        print(x.shape)
        im2 = x.copy()
        im2[:,:, :, 0] = x[:, :, :, 2]
        im2[:,:, :, 2] = x[:, :, :, 0]
        x = im2
        
        ANotherImg = y.copy()
        ANotherImg[:, :, :, 0] = y[:, :, :, 2]
        ANotherImg[:, :, :, 2] = y[:, :, :, 0]
        y = ANotherImg
        
        ImageLoader.mkdir_safe(WheretosavePickleData)
        pickle_out = open(WheretosavePickleData + "/" + pickleOutX, "wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()

        pickle_out = open(WheretosavePickleData + "/" + pickleOutY, "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()
        print("Finished saving images to pickle x for " + pickleOutX + "  y for " + pickleOutY)

    def save_to_pickle(array, pickleOut: str, savePath: str):
        ImageLoader.mkdir_safe(savePath)
        pickle_out = open(savePath + "/" + pickleOut, "wb")
        pickle.dump(array, pickle_out)
        pickle_out.close()

    def load_from_pickle(loadPath: str, fileName: str):
         pickle_in = open(loadPath + "/" + fileName, "rb")
         return pickle.load(pickle_in)

    def load_array_from_pickle(loadPath: str, inX, inY):
        pickle_in = open(loadPath + "/" + inX, "rb")
        x_train = pickle.load(pickle_in)
        
        pickle_in = open(loadPath + "/" + inY, "rb")
        y_train = pickle.load(pickle_in)
        return [x_train, y_train]

    def combine_images(imageArray, max_y: int, max_x: int) -> np.array:
        image = np.zeros((max_y,max_x,3),np.int)
        #np.array(x).reshape(-1,resize[1],resize[2],3)
        size_y, size_x,Nope = imageArray[0].shape
        curr_y = 0
        i = 0
      
        while (curr_y + size_y) <= max_y:
            curr_x = 0
            while (curr_x + size_x) <= max_x:
                try:
                    image[curr_y:curr_y + size_y, curr_x:curr_x + size_x] = imageArray[i]
                except:
                    i -= 1
                i += 1
                curr_x += size_x
            curr_y += size_y
        return image


    def remove_dark_images(img, gt):
        nrOfDarkImages = 0
        dimension = 0
        newimg = []
        newgt = [] 
        
        for i in range(len(gt)):
            thisarray = np.asarray(gt[i])
            arrayflow = np.all(thisarray < 2, dimension)
            if(arrayflow.all()):
                nrOfDarkImages = nrOfDarkImages + 1
            else:
                newimg.append(img[i])
                newgt.append(gt[i])
        #print(nrOfDarkImages)
        return [newimg,newgt]

    
    def add_border(array, dimensionX, dimensionY, borderColor, max_x, max_y):
        # Takes in the prefer Images and adds a border if nessesary based on dimensions
        border_y = 0
        if max_y % dimensionY != 0:
            border_y = (dimensionY - (max_y % dimensionY) + 1) // 2
            array = cv2.copyMakeBorder(array, border_y, border_y, 0, 0, 
                                       cv2.BORDER_CONSTANT, value=borderColor)
        border_x = 0
        if max_x % dimensionX != 0:
            border_x = (dimensionX - (max_x % dimensionX) + 1) // 2
            array = cv2.copyMakeBorder(array, 0, 0, border_x, border_x, 
                                       cv2.BORDER_CONSTANT, value=borderColor)
        return array
        
    def split_image(array, dimensionX, dimensionY, max_x, max_y):
        chunks = []
        current_y = 0
        while(current_y + dimensionY)<= max_y:
            current_x = 0
            while(current_x + dimensionX) <= max_x:
                chunks.append(array[current_y:current_y + dimensionY, current_x : current_x + dimensionX])
                current_x += dimensionX
            current_y += dimensionY
        return chunks

    def save_image(path: str, image: [np.array], name: str):
        ImageLoader.mkdir_safe(path)
        cv2.imwrite(os.path.join(path, name + '.png'), image)
    
    def mkdir_safe(path: str):
        if not os.path.exists(path):
            os.makedirs(path)

    def transform_list_to_combined_np(array: list):
        x = [] #TrainImage
        y = [] #TrainimageResult
        
        for startPicture, groundTruth in array:
            x.append(startPicture)
            y.append(groundTruth)
            
        x = np.array(x)
        y = np.array(y)
        return [x, y]

    def convert_list_to_np(array: list):
        numpyArray = [] 
        for eachArrayElement in array:
            numpyArray.append(eachArrayElement)
        numpyArray = np.array(numpyArray)
        return numpyArray

    def adjust_colors(array : np.array):
        copy = array.copy()
        copy[:,:, :, 0] = array[:, :, :, 2]
        copy[:,:, :, 2] = array[:, :, :, 0]
        return copy
    
    def flatten_data_multi(x_train, y_train):
        x_train = x_train.reshape(y_train.shape[0], conf.Xsize*conf.Ysize*3)
        x_train = x_train.astype('float32') / 255
        y_train = y_train.reshape(y_train.shape[0], conf.Xsize*conf.Ysize*3)
        y_train = y_train.astype('float32') / 255
        return x_train, y_train
    
    def flatten_data(x_train):
        x_train = x_train.reshape(x_train.shape[0], conf.Xsize*conf.Ysize*3)
        x_train = x_train.astype('float32') / 255
        return x_train
        
    def convert_to_multidimensional_data_multi(x_train, y_train):
        x_train = x_train.reshape(x_train.shape[0], conf.Xsize,conf.Ysize, 3)
        x_train = x_train.astype('float32') * 255
        y_train = y_train.reshape(y_train.shape[0], conf.Xsize,conf.Ysize, 3)
        y_train = y_train.astype('float32') * 255
        return x_train, y_train
    
    def convert_to_multidimensional_data(x_train):
        x_train = x_train.reshape(x_train.shape[0], conf.Xsize,conf.Ysize, 3)
        x_train = x_train.astype('float32') * 255
        return x_train

    def reEachLabelGt(Array: np.array):
        returnMe = []
        Togglefunction =  False
        for color_data in Array:
            Togglefunction =  False
            if(color_data[0] == 128):
                if(color_data[2] == 1):
                    Togglefunction = True
                    returnMe.append([1,1,0,0,0])
                elif (color_data[2] == 2):
                    Togglefunction = True
                    returnMe.append([1,0,1,0,0])
                elif (color_data[2] == 3):
                    Togglefunction = True
                    returnMe.append([1,1,1,0,0])
                elif (color_data[2] == 4):
                    Togglefunction = True
                    returnMe.append([1,0,0,1,0])
                elif (color_data[2] == 5):
                    Togglefunction = True
                    returnMe.append([1,1,0,1,0])
                elif (color_data[2] == 6):
                    Togglefunction = True
                    returnMe.append([1,0,1,1,0])
                elif (color_data[2] == 8):
                    Togglefunction = True
                    returnMe.append([1,0,0,0,1])
                elif (color_data[2] == 9):
                    Togglefunction = True
                    returnMe.append([1,1,0,0,1])
                elif (color_data[2] == 10):
                    Togglefunction = True
                    returnMe.append([1,0,1,0,1])
                elif (color_data[2] == 12):
                    Togglefunction = True
                    returnMe.append([1,0,0,1,1])
            elif (color_data[0] == 0):
                if(color_data[2] == 1):
                    Togglefunction = True
                    returnMe.append([0,1,0,0,0])
                elif (color_data[2] == 2):
                    Togglefunction = True
                    returnMe.append([0,0,1,0,0])
                elif (color_data[2] == 3):
                    Togglefunction = True
                    returnMe.append([0,1,1,0,0])
                elif (color_data[2] == 4):
                    Togglefunction = True
                    returnMe.append([0,0,0,1,0])
                elif (color_data[2] == 5):
                    Togglefunction = True
                    returnMe.append([0,1,0,1,0])
                elif (color_data[2] == 6):
                    Togglefunction = True
                    returnMe.append([0,0,1,1,0])
                elif (color_data[2] == 8):
                    Togglefunction = True
                    returnMe.append([0,0,0,0,1])
                elif (color_data[2] == 9):
                    Togglefunction = True
                    returnMe.append([0,1,0,0,1])
                elif (color_data[2] == 10):
                    Togglefunction = True
                    returnMe.append([0,0,1,0,1])
                elif (color_data[2] == 12):
                    Togglefunction = True
                    returnMe.append([0,0,0,1,1]) 
                elif (color_data[2] == 0):
                    Togglefunction = True
                    returnMe.append([0,0,0,0,0])
            if(Togglefunction == False):
                print("these values are not labeled should do?"+str(color_data[0])+","+str(color_data[1])+","+str(color_data[2]))
        return returnMe

    def reLabelGt(Array: np.array):
        returnThisArray = []
        for each_array in Array:
            returnThisArray.append(ImageLoader.reEachLabelGt(each_array))
        return returnThisArray


    def turnLabeltoColorvalues(Array):
        returnThisArray = []
        for each_array in Array:
            returnThisArray.append(ImageLoader.turnlabeltocolorsingle(each_array))
        return returnThisArray

    def turnlabeltocolorsingle(Array: np.array):
        returnMe = []
        Togglefunction =  False
        for label_data in Array:
            Togglefunction =  False
            if((label_data == [1,1,0,0,0]).all()):
                Togglefunction = True
                returnMe.append([128,0,1])
            elif(((label_data == [1,0,1,0,0])).all()):
                Togglefunction = True
                returnMe.append([128,0,2])
            elif(((label_data == [1,1,1,0,0])).all()):
                Togglefunction = True
                returnMe.append([128,0,3])
            elif(((label_data == [1,0,0,1,0])).all()):
                Togglefunction = True
                returnMe.append([128,0,4])
            elif(((label_data == [1,1,0,1,0])).all()):
                Togglefunction = True
                returnMe.append([128,0,5])
            elif(((label_data == [1,0,1,1,0])).all()):
                Togglefunction = True
                returnMe.append([128,0,6])
            elif(((label_data == [1,0,0,0,1])).all()):
                Togglefunction = True
                returnMe.append([128,0,8])
            elif(((label_data == [1,1,0,0,1])).all()):
                Togglefunction = True
                returnMe.append([128,0,9])
            elif(((label_data == [1,0,1,0,1])).all()):
                Togglefunction = True
                returnMe.append([128,0,10])
            elif(((label_data == [1,0,0,1,1])).all()):
                Togglefunction = True
                returnMe.append([128,0,12])
########################################################
            elif((label_data == [0,1,0,0,0]).all()):
                Togglefunction = True
                returnMe.append([0,0,1])
            elif(((label_data == [0,0,1,0,0])).all()):
                Togglefunction = True
                returnMe.append([0,0,2])
            elif(((label_data == [0,1,1,0,0])).all()):
                Togglefunction = True
                returnMe.append([0,0,3])
            elif(((label_data == [0,0,0,1,0])).all()):
                Togglefunction = True
                returnMe.append([0,0,4])
            elif(((label_data == [0,1,0,1,0])).all()):
                Togglefunction = True
                returnMe.append([0,0,5])
            elif(((label_data == [0,0,1,1,0])).all()):
                Togglefunction = True
                returnMe.append([0,0,6])
            elif(((label_data == [0,0,0,0,1])).all()):
                Togglefunction = True
                returnMe.append([0,0,8])
            elif(((label_data == [0,1,0,0,1])).all()):
                Togglefunction = True
                returnMe.append([0,0,9])
            elif(((label_data == [0,0,1,0,1])).all()):
                Togglefunction = True
                returnMe.append([0,0,10])
            elif(((label_data == [0,0,0,1,1])).all()):
                Togglefunction = True
                returnMe.append([0,0,12])
            elif(((label_data == [0,0,0,0,0])).all()):
                Togglefunction = True
                returnMe.append([0,0,0])
            if(Togglefunction == False):
                returnMe.append([0,128,0])
                #print("these values are not labeled should do?")
        return returnMe

def main():
    # read all data from folders and add border if needed. Afterwards split images into chunks
    data = ImageLoader.read_images_resize_crop(conf.DATADIR, conf.Training, conf.Result,
                                               [0,0,0], [1, conf.Xsize, conf.Ysize])
    img = []
    gt = []
    
    for eachArrayElement in data:
        img.append(eachArrayElement[0])
        gt.append(eachArrayElement[1])
    
    # convert to numpy arrays
    img = ImageLoader.convert_list_to_np(img)
    gt = ImageLoader.convert_list_to_np(gt)
    gt = ImageLoader.adjust_colors(gt)
    img = ImageLoader.adjust_colors(img)
    #print(gt.shape)
    
    # write original image to pickle
    ImageLoader.save_to_pickle(img, "combined.pickle", conf.Picklefiles)
    gt = gt.reshape(gt.shape[0], conf.Xsize*conf.Ysize*3)
    img = img.reshape(img.shape[0], conf.Xsize*conf.Ysize*3)
    
    # remove complete dark images in the image
#    [img,gt] = ImageLoader.remove_dark_images(img, gt)
    img = ImageLoader.convert_list_to_np(img)
    gt = ImageLoader.convert_list_to_np(gt)
    gt = gt.reshape(gt.shape[0], conf.Xsize*conf.Ysize, 3)
    gt = ImageLoader.reLabelGt(gt)
    gt = ImageLoader.convert_list_to_np(gt)
    gt = gt.reshape(gt.shape[0], conf.Xsize, conf.Ysize, 5)
    # output training data to pickle
    img = img.reshape(img.shape[0], conf.Xsize, conf.Ysize, 3)
    ImageLoader.save_to_pickle(img, "img.pickle", conf.Picklefiles)
    ImageLoader.save_to_pickle(gt, "gt.pickle", conf.Picklefiles)
    #print(gt.shape)
#    plt.imshow(gt[1])
#    plt.show()
    
if __name__ == "__main__":
    main()