import numpy as np
import os
import cv2
import utils.config as conf 
from numba import guvectorize
from numba import int64
from timeit import default_timer as timer

@guvectorize([(int64[:], int64[:], int64[:])], '(n),(i)->(i)', target='cpu')
def turnlabeltoColorSingleCuda(label_data, array):
    
    # some sequential steps are missing...?
    if((label_data[0] != 0)):
        if((label_data == [1,1,0,0,0]).all()):
            array = relabel_colors(array, 128, 0, 1)
        elif(((label_data == [1,0,1,0,0])).all()):
            array = relabel_colors(array, 128, 0, 2)
        elif(((label_data == [1,1,1,0,0])).all()):
            array = relabel_colors(array, 128, 0, 3)
        elif(((label_data == [1,0,0,1,0])).all()):
            array = relabel_colors(array, 128, 0, 4)
        elif(((label_data == [1,1,0,1,0])).all()):
            array = relabel_colors(array, 128, 0, 5)
        elif(((label_data == [1,0,1,1,0])).all()):
            array = relabel_colors(array, 128, 0, 6)
        elif(((label_data == [1,0,0,0,1])).all()):
            array = relabel_colors(array, 128, 0, 8)
        elif(((label_data == [1,1,0,0,1])).all()):
            array = relabel_colors(array, 128, 0, 9)
        elif(((label_data == [1,0,1,0,1])).all()):
            array = relabel_colors(array, 128, 0, 10)
        elif(((label_data == [1,0,0,1,1])).all()):
            array = relabel_colors(array, 128, 0, 12)
    elif(((label_data == [0,0,0,0,0])).all()):
        array = relabel_colors(array, 0, 0, 0)
    elif((label_data == [0,1,0,0,0]).all()):
        array = relabel_colors(array, 0, 0, 1)
    elif(((label_data == [0,0,1,0,0])).all()):
        array = relabel_colors(array, 0, 0, 2)
    elif(((label_data == [0,1,1,0,0])).all()):
        array = relabel_colors(array, 0, 0, 3)
    elif(((label_data == [0,0,0,1,0])).all()):
        array = relabel_colors(array, 0, 0, 4)
    elif(((label_data == [0,1,0,1,0])).all()):
        array = relabel_colors(array, 0, 0, 5)
    elif(((label_data == [0,0,1,1,0])).all()):
        array = relabel_colors(array, 0, 0, 6)
    elif(((label_data == [0,0,0,0,1])).all()):
        array = relabel_colors(array, 0, 0, 7)
    elif(((label_data == [0,1,0,0,1])).all()):
        array = relabel_colors(array, 0, 0, 9)
    elif(((label_data == [0,0,1,0,1])).all()):
        array = relabel_colors(array, 0, 0, 10)
    elif(((label_data == [0,0,0,1,1])).all()):
        array = relabel_colors(array, 0, 0, 12)
    else:
        array = relabel_colors(array, 0, 128, 0)

@guvectorize([(int64[:],int64[:], int64[:])], '(n),(i)->(i)')
def reEachLabelGtCuda(color_data,justforSize, returnMe):
    if(color_data[0] == 128):
        if(color_data[2] == 1):
            returnMe[0] = 1
            returnMe[1] = 1
            returnMe[2] = 0
            returnMe[3] = 0
            returnMe[4] = 0
        elif (color_data[2] == 2):
            returnMe[0] = 1
            returnMe[1] = 0
            returnMe[2] = 1
            returnMe[3] = 0
            returnMe[4] = 0
        elif (color_data[2] == 3):
            returnMe[0] = 1
            returnMe[1] = 1
            returnMe[2] = 1
            returnMe[3] = 0
            returnMe[4] = 0
        elif (color_data[2] == 4):
            returnMe[0] = 1
            returnMe[1] = 0
            returnMe[2] = 0
            returnMe[3] = 1
            returnMe[4] = 0
        elif (color_data[2] == 5):
            returnMe[0] = 1
            returnMe[1] = 1
            returnMe[2] = 0
            returnMe[3] = 1
            returnMe[4] = 0
        elif (color_data[2] == 6):
            returnMe[0] = 1
            returnMe[1] = 0
            returnMe[2] = 1
            returnMe[3] = 1
            returnMe[4] = 0
        elif (color_data[2] == 8):
            returnMe[0] = 1
            returnMe[1] = 0
            returnMe[2] = 0
            returnMe[3] = 0
            returnMe[4] = 1
        elif (color_data[2] == 9):
            returnMe[0] = 1
            returnMe[1] = 1
            returnMe[2] = 0
            returnMe[3] = 0
            returnMe[4] = 1
        elif (color_data[2] == 10):
            returnMe[0] = 1
            returnMe[1] = 0
            returnMe[2] = 1
            returnMe[3] = 0
            returnMe[4] = 1
        elif (color_data[2] == 12):
            returnMe[0] = 1
            returnMe[1] = 0
            returnMe[2] = 0
            returnMe[3] = 1
            returnMe[4] = 1
    elif (color_data[0] == 0):
        if(color_data[2] == 1):
            returnMe[0] = 0
            returnMe[1] = 1
            returnMe[2] = 0
            returnMe[3] = 0
            returnMe[4] = 0
        elif (color_data[2] == 2):
            returnMe[0] = 0
            returnMe[1] = 0
            returnMe[2] = 1
            returnMe[3] = 0
            returnMe[4] = 0
        elif (color_data[2] == 3):
            returnMe[0] = 0
            returnMe[1] = 1
            returnMe[2] = 1
            returnMe[3] = 0
            returnMe[4] = 0
        elif (color_data[2] == 4):
            returnMe[0] = 0
            returnMe[1] = 0
            returnMe[2] = 0
            returnMe[3] = 1
            returnMe[4] = 0
        elif (color_data[2] == 5):
            returnMe[0] = 0
            returnMe[1] = 1
            returnMe[2] = 0
            returnMe[3] = 1
            returnMe[4] = 0
        elif (color_data[2] == 6):
            returnMe[0] = 0
            returnMe[1] = 0
            returnMe[2] = 1
            returnMe[3] = 1
            returnMe[4] = 0
        elif (color_data[2] == 8):
            returnMe[0] = 0
            returnMe[1] = 0
            returnMe[2] = 0
            returnMe[3] = 0
            returnMe[4] = 1
        elif (color_data[2] == 9):
            returnMe[0] = 0
            returnMe[1] = 1
            returnMe[2] = 0
            returnMe[3] = 0
            returnMe[4] = 1
        elif (color_data[2] == 10):
            returnMe[0] = 0
            returnMe[1] = 0
            returnMe[2] = 1
            returnMe[3] = 0
            returnMe[4] = 1
        elif (color_data[2] == 12):
            returnMe[0] = 0
            returnMe[1] = 0
            returnMe[2] = 0
            returnMe[3] = 1
            returnMe[4] = 1
        elif (color_data[2] == 0):
            returnMe[0] = 0
            returnMe[1] = 0
            returnMe[2] = 0
            returnMe[3] = 0
            returnMe[4] = 0
            
def relabel_colors(array, r, g, b):      
    array[0] = r
    array[1] = g
    array[2] = b
    return array

class ImageLoader():
    """Load dataset images, split them inte chunks and write them to pickle files."""
    


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

    def convert_list_to_np(array: list):
        numpyArray = np.array(array)
        return numpyArray

    def adjust_colors(array : np.array):
        copy = array.copy()
        value = len(array.shape)
        if(len(array.shape)==4):           
            copy[:,:, :, 0] = array[:, :, :, 2]
            copy[:,:, :, 2] = array[:, :, :, 0]
            return copy
        else:
            copy[:,:, 0] = array[:, :, 2]
            copy[:,:, 2] = array[:, :, 0]
            return copy

    def reLabelGt(Array: np.array):
        returnThisArray = []
        b = [[1,2,3,4,5]]
        for each_array in Array:
            returnThisArray.append(reEachLabelGtCuda(each_array,b))
        return returnThisArray

    def turnLabeltoColorvalues(Array):
        returnThisArray = []
        b = [[1,2,3]]
        returnThisArray.append(turnlabeltoColorSingleCuda(Array,b))
        return returnThisArray

    def read_Images(*args):
        dataDir = args[0]
        trainingDir = args[1]
        resultDir = args[2]
        crop = args[3]
        resize = [0,0,0]
        if len(args) > 4:
            resize = args[4]
        #elif len(args) == 1:

        returnImg = []
        returnGt = []
        #training_data = []
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

                    imgTraining_array = ImageLoader.adjust_colors(imgTraining_array)
                    imgResult_array = ImageLoader.adjust_colors(imgResult_array)
                    b = [[1,2,3,4,5]]
                    imgResult_array=reEachLabelGtCuda(imgResult_array,b)
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
                        returnImg+=(imgTraining_array)
                        returnGt+=(imgResult_array)
                        #training_data += list(zip(imgTraining_array, imgResult_array))
                    else:
                        img.append(imgTraining_array)
                        gt.append(imgResult_array)
                        #training_data.append([imgTraining_array, imgResult_array])
                    i= i+1
                except Exception as e:
                    pass
        return [returnImg,returnGt]

    def shortMain():
        # read all data from folders and add border if needed. Afterwards split images into chunks
        start = timer()
        [img,gt] =ImageLoader.read_Images(conf.DATADIR, conf.Training, conf.Result, [1, conf.Xsize, conf.Ysize])


        totaltime = timer() - start
        print("loaded all pictures 1 and it took "+(str)(totaltime))
        # write original image to pickle
        [PredictPictures,GTPredictPictures]=ImageLoader.read_Images(conf.DATADIR,conf.TestTraining,conf.TestResult,[1, conf.Xsize, conf.Ysize])
        print("2")
        # remove complete dark images in the image
        #gt = gt.reshape(gt.shape[0], conf.Xsize*conf.Ysize*3)
        #img = img.reshape(img.shape[0], conf.Xsize*conf.Ysize*3)
        #[img,gt] = ImageLoader.remove_dark_images(img, gt)

        img = ImageLoader.convert_list_to_np(img)
        gt = ImageLoader.convert_list_to_np(gt)

        gt = gt.reshape(gt.shape[0], conf.Xsize, conf.Ysize, 5)
        # output training data to pickle
        img = img.reshape(img.shape[0], conf.Xsize, conf.Ysize, 3)
        totaltime = timer() - start
        print("total tid förall laddning"+ (str)(totaltime))        
        return [img,gt,PredictPictures,GTPredictPictures]

def main():
    # empty for now
    
if __name__ == "__main__":
    main()