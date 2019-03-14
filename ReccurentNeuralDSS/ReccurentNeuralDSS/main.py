import matplotlib.pyplot as plt
from utils.imageLoader import ImageLoader
from model.model import Model
import utils.config as conf
import numpy as np
from keras.callbacks import TensorBoard

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


    xreshapeValue = [conf.Xsize,conf.Ysize*3] # for LSTMRNN
    yreshapeValue = [conf.Xsize,conf.Ysize*5] # for LSTMRNN
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
    elif len(args) > 3:
        x_train = args[0]
        y_train = args[1]

    yreshapeValue = [conf.Xsize,conf.Ysize*5] 
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    model = Model.build_CNN_model(x_train[0].shape);
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,validation_split=conf.validationSplit)
    Model.model = model;
    return False

def newCNNBIDirectionalLstmRNN(*args):
    # create log dir if not created when executed, then when the model is done with the training,
    # open cmd and locate to the directory above logs and run: "tensorboard --logdir=logs/"
    # then copy the url into the browser to visualize results
    tensorboard = TensorBoard(log_dir='logs/')
    
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

    yreshapeValue = [conf.Xsize,conf.Ysize*5] # for LSTMRNN
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0]*yreshapeValue[1])
    model = Model.build_CNN_model(x_train[0].shape);
    model.fit(x_train, y_train, epochs=conf.AmountOfEpochs, batch_size=conf.batchSize,
              validation_split=conf.validationSplit, callbacks=[tensorboard])
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
        i = i+1
        #if(i%1000 == 0):
        print(i)

    prediction=prediction.reshape(validation.shape[0],conf.Xsize, conf.Ysize,5)
    prediction = newPrediction
    prediction = ImageLoader.convert_list_to_np(prediction)
    prediction = prediction.reshape(predictionorignalshape, conf.Xsize,conf.Ysize, 3)    
    # show result
    if(Number <10):
        img = ImageLoader.combine_images(prediction, 4872, 6496)
    elif(Number <40):
        img = ImageLoader.combine_images(prediction, 3328, 4992)
    img = ImageLoader.adjust_colors(img)
    Number = ""
    if(len(args)==3):
        Number = str(args[2])
    ImageLoader.save_image(conf.WhereTosaveTestImage,img,conf.NameOfTestImage + str(Number))
#    plt.imshow(img)
#    plt.show()
    return 0

def main():
    [x_train,y_train,PredictPictures,CorrectPrediction]=ImageLoader.shortMain()
    newCNNBIDirectionalLstmRNN(x_train,y_train)
#    Model.save_model("CNNBID")
    #Model.model = Model.load_model("CNNBID")
    i = 0
    for eachPicture in PredictPictures:
        SaveImage(False, eachPicture,i)
        i = i+1
    

if __name__ == "__main__":
    main()
