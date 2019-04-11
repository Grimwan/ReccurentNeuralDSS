import matplotlib.pyplot as plt
from utils.imageLoader import ImageLoader
from model.model import Model
from model.model import Unet
import utils.config as conf
import numpy as np
from keras.callbacks import TensorBoard
import tensorflow as tf
import keras as K



def BiDirectionalLSTMRNN(*args):

    loadModel = ""
    if len(args) == 1:
        x_train = args[0]
        y_train = args[0]
    elif len(args) == 2:
        x_train = args[0]
        y_train = args[1]
    elif len(args) > 2:
        x_train = args[0]
        y_train = args[1]
        loadModel = args[2]
    elif len(args) > 3:
        x_train = args[0]
        y_train = args[1]
        x_validation = args[2]
        y_validation = args[3]
             

    xreshapeValue = [conf.Xsize,conf.Ysize*3] # for LSTMRNN
    yreshapeValue = [conf.Xsize,conf.Ysize*5] # for LSTMRNN
    x_train = x_train.reshape(x_train.shape[0], xreshapeValue[0],xreshapeValue[1])
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0],yreshapeValue[1])
    if(loadModel ==""):
        model = Model.build_BI_LSTM_model(x_train[0].shape)
    else:
        model = Model.load_model(loadModel)
        model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,
              validation_split=conf.validationSplit)
    Model.model = model;
    return True

def StandardCNN(*args):
     
    if len(args) == 1:
        x_train = args[0]
        y_train = args[0]
    elif len(args) == 2:
        x_train = args[0]
        y_train = args[1]
    elif len(args) == 3:
        x_train = args[0]
        y_train = args[1]
    elif len(args) > 3:
        x_train = args[0]
        y_train = args[1]

    yreshapeValue = [conf.Xsize,conf.Ysize*5] 
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    #model = Model.build_CNN_model(x_train[0].shape);
    model = Model.build_Unet_model(x_train[0].shape)
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)
    Model.model = model;
    return False

def CNNBIDirectionalLstmRNN(*args):
    # create log dir if not created when executed, then when the model is done with the training,
    # open cmd and locate to the directory above logs and run: "tensorboard --logdir=logs/"
    # then copy the url into the browser to visualize results
    tensorboard = TensorBoard(log_dir='logs/')
    loadModel = ""
    if len(args) < 2:
        x_train = args[0]
        y_train = args[0]
    elif len(args) < 3:
        x_train = args[0]
        y_train = args[1]
    elif len(args) < 4:
        x_train = args[0]
        y_train = args[1]
        loadModel = args[2]
    yreshapeValue = [conf.Xsize,conf.Ysize*5] # for LSTMRNN
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    if(loadModel ==""):
        model = Model.build_CNN_BI_LSTM_model(x_train[0].shape)
    else:
        model = Model.load_model(loadModel)
        model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,
              validation_split=conf.validationSplit, callbacks=[tensorboard])
    Model.model = model;
    return False


def ReNet(*args):
    # create log dir if not created when executed, then when the model is done with the training,
    # open cmd and locate to the directory above logs and run: "tensorboard --logdir=logs/"
    # then copy the url into the browser to visualize results
    tensorboard = TensorBoard(log_dir='logs/')
    loadModel = ""
    if len(args) < 2:
        x_train = args[0]
        y_train = args[0]
    elif len(args) < 3:
        x_train = args[0]
        y_train = args[1]
    elif len(args) < 4:
        x_train = args[0]
        y_train = args[1]
        loadModel = args[2]
    yreshapeValue = [conf.Xsize,conf.Ysize*5] # for LSTMRNN
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    if(loadModel ==""):
        model = Model.ReNet(x_train[0].shape)
    else:
        model = Model.load_model(loadModel)
        model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,
              validation_split=conf.validationSplit, callbacks=[tensorboard])
    Model.model = model;
    return False



def TrainNetwork(*args):
    if(len(args)>3):
        if(args[0] == 0):
            BiDirectionalLSTMRNN(args[1],args[2],args[3])
        elif(args[0] == 1):
            StandardCNN(args[1],args[2],args[3])
        elif(args[0] == 2):
            CNNBIDirectionalLstmRNN(args[1],args[2],args[3])
        elif(args[0] ==3):
            ReNet(args[1],args[2],args[3])
    else:
        if(args[0] == 0):
            BiDirectionalLSTMRNN(args[1],args[2])
        elif(args[0] == 1):
            StandardCNN(args[1],args[2])
        elif(args[0] == 2):
            CNNBIDirectionalLstmRNN(args[1],args[2])
        elif(args[0] ==3):
            ReNet(args[1],args[2])


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
        i = i+1
        #if(i%1000 == 0):
        print(i)

    prediction=prediction.reshape(validation.shape[0],conf.Xsize, conf.Ysize,5)
    prediction = newPrediction
    prediction = ImageLoader.convert_list_to_np(prediction)
    prediction = prediction.reshape(predictionorignalshape, conf.Xsize,conf.Ysize, 3)    
    # show result

    img = ImageLoader.combine_images(prediction, args[3], args[4])
    img = ImageLoader.adjust_colors(img)
    Number = str(args[2])
    ImageLoader.save_image(conf.WhereTosaveTestImage,img,conf.NameOfTestImage + str(Number))
#    plt.imshow(img)
#    plt.show()
    return 0


def Train(*args):
    NN = 0
    Predict = True
    WhereToSave = "CNNBID"
    if(len(args)>0):
        if(args[0] == "BiDirectionalLSTMRNN"):
            NN = 0
        elif(args[0] == "StandardCNN"):
            NN = 1
        elif(args[0] == "CNNBIDirectionalLstmRNN"):
            NN = 2
        elif(args[0] == "ReNet"):
            NN = 3
        if(len(args)>1):
            Predict = args[1]

    #[x_trainCB55,y_trainCB55,x_trainCS18,y_trainCS18,x_trainCS863,y_trainCS863,PredictionPictureCB55,PredictionPictureCS]=ImageLoader.shortMain()
    [x_train,y_train] = ImageLoader.read_Images(conf.DATADIR,["CB55/img/training"],  ["CB55/pixel-level-gt/training"], [1, conf.Xsize, conf.Ysize],True,True)    
    #[x_trainCB55,y_trainCB55] = ImageLoader.augment_Images(x_trainCB55,y_trainCB55);
    TrainNetwork(NN,x_train,y_train)
    Model.save_model("CNNBID")
    Model.model = None
    tf.keras.backend.clear_session()
    K.backend.clear_session()
    [x_train,y_train] = ImageLoader.read_Images(conf.DATADIR,["CS18/img/training"],  ["CS18/pixel-level-gt/training"], [1, conf.Xsize, conf.Ysize],True,True)
    TrainNetwork(NN,x_train,y_train,"CNNBID")
    Model.save_model("CNNBID")
    [x_train,y_train] = ImageLoader.read_Images(conf.DATADIR,["CS863/img/training"],  ["CS863/pixel-level-gt/training"], [1, conf.Xsize, conf.Ysize],True,True)
    TrainNetwork(NN,x_train,y_train,"CNNBID")

    #Model.model = Model.load_model("CNNBID")
    i = 0
    Model.save_model("CNNBID")
    [PredictionPictureCB55,GTPredictPicturesCB55]=ImageLoader.read_Images(conf.DATADIR,conf.PredictionPictureCB55,conf.PredictionPictureResultCB55,[1, conf.Xsize, conf.Ysize],False)
    for eachPicture in PredictionPictureCB55:
        SaveImage(False, eachPicture,i,6496,4872)
        i = i+1
    del PredictionPictureCB55
    PredictionPictureCB55 = []
    del GTPredictPicturesCB55
    GTPredictPicturesCB55 = []
    [PredictPicturesCS,GTPredictPicturesCS]=ImageLoader.read_Images(conf.DATADIR,conf.PredictionPictureCS,conf.PredictionPictureResultCS,[1, conf.Xsize, conf.Ysize],False)
    for eachPicture in PredictionPictureCS:
        SaveImage(False, eachPicture,i,4992,3328)
        i = i+1
    del PredictPicturesCS
    del GTPredictPicturesCS
    PredictPicturesCS =  []
    GTPredictPicturesCS = []

def ExperimentTraining(*args):
    NN = 0
    Predict = True
    WhereToSave = "CNNBID"
    if(len(args)>0):
        if(args[0] == "BiDirectionalLSTMRNN"):
            NN = 0
        elif(args[0] == "StandardCNN"):
            NN = 1
        elif(args[0] == "CNNBIDirectionalLstmRNN"):
            NN = 2
        if(len(args)>1):
            Predict = args[1]

    #Model.ReNet([32,32,3])
    #Model.build_CNN_BI_LSTM_model([64,64,3])