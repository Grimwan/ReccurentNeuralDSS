import matplotlib.pyplot as plt
from utils.imageLoader import ImageLoader
from model.model import Model
import utils.config as conf
from keras import preprocessing
import numpy as np

def BiDirectionalLSTMRNN(x_train,y_train):
    xreshapeValue = [conf.Xsize,conf.Ysize*3] #for LSTMRNN
    yreshapeValue = [conf.Xsize,conf.Ysize*5] #for LSTMRNN
    x_train = x_train.reshape(x_train.shape[0], xreshapeValue[0],xreshapeValue[1])
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0],yreshapeValue[1])

    model = Model.build_BI_LSTM_model(x_train[0].shape);
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)
    
    if(SaveTestImage):
        validation = ImageLoader.load_from_pickle(conf.Picklefiles, conf.LoadTestPickle)
        validation = validation.reshape(validation.shape[0], xreshapeValue[0],xreshapeValue[1])
        validation = validation.astype('float32') / 255
        prediction = model.predict(validation)
        prediction = np.where(prediction <= 0.5, 0, 1)
        print(prediction.shape)
        6964#30856
        prediction=prediction.reshape(30856,yreshapeValue[0], 32,5)
        i = 0
        newPrediction = []
        for eachprediction in prediction:
            newPrediction.append(ImageLoader.turnLabeltoColorvalues(eachprediction))
            i= i+1
        prediction = newPrediction
        prediction = ImageLoader.convert_list_to_np(prediction)
        prediction = prediction.reshape(prediction.shape[0], conf.Xsize,conf.Ysize, 3)    
        # show result
        img = ImageLoader.combine_images(prediction, 6496, 4872)
        img = ImageLoader.adjust_colorsthree(img)
        ImageLoader.save_image(conf.WhereTosaveTestImage,img,conf.NameOfTestImage)
        plt.imshow(img)
        plt.show()

    return 0;

def StandardCNN(x_train,y_train):
    yreshapeValue = [conf.Xsize,conf.Ysize*5] 
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    model = Model.build_CNN_model(x_train[0].shape);
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)

    if(SaveTestImage):
        validation = ImageLoader.load_from_pickle(conf.Picklefiles, conf.LoadTestPickle)
        validation = validation.astype('float32') / 255
        prediction = model.predict(validation)
        prediction = np.where(prediction <= 0.5, 0, 1)
        print(prediction.shape)
        6964#30856
        prediction=prediction.reshape(30856,yreshapeValue[0], 32,5)
        i = 0
        newPrediction = []
        for eachprediction in prediction:
            newPrediction.append(ImageLoader.turnLabeltoColorvalues(eachprediction))
            i= i+1
        prediction = newPrediction
        prediction = ImageLoader.convert_list_to_np(prediction)
        prediction = prediction.reshape(prediction.shape[0], conf.Xsize,conf.Ysize, 3)    
        # show result
        img = ImageLoader.combine_images(prediction, 6496, 4872)
        img = ImageLoader.adjust_colorsthree(img)
        ImageLoader.save_image(conf.WhereTosaveTestImage,img,conf.NameOfTestImage)
        plt.imshow(img)
        plt.show()
    return 0;

def CNNBIDirectionalLstmRNN(x_train,y_train):
    xreshapeValue = [conf.Xsize,conf.Ysize*3] #for LSTMRNN
    yreshapeValue = [conf.Xsize,conf.Ysize*5] #for LSTMRNN
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    model = Model.build_CNN_model(x_train[0].shape);
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)
    
    Model.model = model;
    Model.save_model();


    if(SaveTestImage):
        validation = ImageLoader.load_from_pickle(conf.Picklefiles, conf.LoadTestPickle)
        validation = validation.astype('float32') / 255
        prediction = model.predict(validation)
        prediction = np.where(prediction <= 0.5, 0, 1)
        print(prediction.shape)
        6964#30856
        prediction=prediction.reshape(30856,yreshapeValue[0], 32,5)
        i = 0
        newPrediction = []
        for eachprediction in prediction:
            newPrediction.append(ImageLoader.turnLabeltoColorvalues(eachprediction))
            i= i+1
        prediction = newPrediction
        prediction = ImageLoader.convert_list_to_np(prediction)
        prediction = prediction.reshape(prediction.shape[0], conf.Xsize,conf.Ysize, 3)    
        # show result
        img = ImageLoader.combine_images(prediction, 6496, 4872)
        img = ImageLoader.adjust_colorsthree(img)
        ImageLoader.save_image(conf.WhereTosaveTestImage,img,conf.NameOfTestImage)
        plt.imshow(img)
        plt.show()

    return 0;
def main():
    x_train = ImageLoader.load_from_pickle(conf.Picklefiles, "img.pickle")
    y_train = ImageLoader.load_from_pickle(conf.Picklefiles, "gt.pickle")

    #BiDirectionalLSTMRNN(x_train,y_train);
    #StandardCNN(x_train,y_train);
    CNNBIDirectionalLstmRNN(x_train,y_train);


 

if __name__ == "__main__":
    main()
