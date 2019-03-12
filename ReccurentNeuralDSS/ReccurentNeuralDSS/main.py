import matplotlib.pyplot as plt
from utils.imageLoader import ImageLoader
from model.model import Model
import utils.config as conf
from keras import preprocessing
import numpy as np

def newBiDirectionalLSTMRNN(*args):

    if len(args) == 1:
        x_train = args[0]
        y_train = args[0]
    elif len(args) == 2:
        x_train = args[0]
        y_train = args[1]
    elif len(args) == 3:
        x_train = args[0]
        y_train = args[1]
        x_validation = args[2]
        y_validation = args[2]
    elif len(args) > 3:
        x_train = args[0]
        y_train = args[1]
        x_validation = args[2]
        y_validation = args[3]


    xreshapeValue = [conf.Xsize,conf.Ysize*3] #for LSTMRNN
    yreshapeValue = [conf.Xsize,conf.Ysize*5] #for LSTMRNN
    x_train = x_train.reshape(x_train.shape[0], xreshapeValue[0],xreshapeValue[1])
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0],yreshapeValue[1])

    model = Model.build_BI_LSTM_model(x_train[0].shape);


    if len(args)>2:
        model.fit(x_train, y_train,x_validation,y_validation, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize)
    else:
        model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)
    Model.model = model;
    return True

def newStandardCNN(*args):
     
    if len(args) == 1:
        x_train = args[0]
        y_train = args[0]
    elif len(args) == 2:
        x_train = args[0]
        y_train = args[1]
    elif len(args) == 3:
        x_train = args[0]
        y_train = args[1]
        x_validation = args[2]
        y_validation = args[2]
    elif len(args) > 3:
        x_train = args[0]
        y_train = args[1]
        x_validation = args[2]
        y_validation = args[3]


    yreshapeValue = [conf.Xsize,conf.Ysize*5] 
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    model = Model.build_CNN_model(x_train[0].shape);
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)
    Model.model = model;
    return False

def newCNNBIDirectionalLstmRNN(*args):
    if len(args) == 1:
        x_train = args[0]
        y_train = args[0]
    elif len(args) == 2:
        x_train = args[0]
        y_train = args[1]
    elif len(args) == 3:
        x_train = args[0]
        y_train = args[1]
        x_validation = args[2]
        y_validation = args[2]
    elif len(args) > 3:
        x_train = args[0]
        y_train = args[1]
        x_validation = args[2]
        y_validation = args[3]


    xreshapeValue = [conf.Xsize,conf.Ysize*3] #for LSTMRNN
    yreshapeValue = [conf.Xsize,conf.Ysize*5] #for LSTMRNN
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    model = Model.build_CNN_model(x_train[0].shape);
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)
    Model.model = model;
    return False

def SaveImage(*args):
    OnlyLstm=args[0]
    validation = ImageLoader.convert_list_to_np(args[1])
    if(OnlyLstm):
        validation = validation.reshape(validation.shape[0],conf.Xsize,conf.Ysize*3)
    validation = validation.astype('float32') / 255
    prediction = Model.model.predict(validation)
    prediction = np.where(prediction <= 0.5, 0, 1)
    print(prediction.shape)
    6964#30856
    predictionorignalshape = prediction.shape[0]
    prediction=prediction.reshape(conf.Xsize, conf.Ysize*validation.shape[0],5)
    i = 0
    newPrediction = []
    for eachprediction in prediction:
        newPrediction.append(ImageLoader.turnLabeltoColorvalues(eachprediction))
        i= i+1
        #if(i%1000 == 0):
        print(i)

    prediction=prediction.reshape(validation.shape[0],conf.Xsize, conf.Ysize,5)
    prediction = newPrediction
    prediction = ImageLoader.convert_list_to_np(prediction)
    prediction = prediction.reshape(predictionorignalshape, conf.Xsize,conf.Ysize, 3)    
    # show result
    img = ImageLoader.combine_images(prediction, conf.orignalPictureX, conf.orignalPictureY)
    img = ImageLoader.adjust_colors(img)
    ImageLoader.save_image(conf.WhereTosaveTestImage,img,conf.NameOfTestImage)
    plt.imshow(img)
    plt.show()
    return 0

def main():
    [x_train,y_train,PredictPictures,CorrectPrediction]=ImageLoader.shortMain()
    newCNNBIDirectionalLstmRNN(x_train,y_train)
#    Model.save_model("CNNBID")
    Model.model = Model.load_model("CNNBID")
    SaveImage(False,PredictPictures);
 

if __name__ == "__main__":
    main()
