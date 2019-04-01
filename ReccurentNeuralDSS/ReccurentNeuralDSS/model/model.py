import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, CuDNNLSTM, Bidirectional, TimeDistributed
from keras import layers
import keras
#lstm: For example, say your input sequences look like X = [[0.54, 0.3], [0.11, 0.2], [0.37, 0.81]]. We can see that this sequence has a timestep of 3 and a data_dim of 2.
class Model:
    """Utility class to represent a model."""
    
    model = 0
       
    def build_BI_LSTM_model(dataSize):
        Model.model = Sequential()
        timesteps = dataSize[0] # first layer array size
        data_dim = dataSize[1] # how big is one part in the array.
        
        # input layer
        Model.model.add(Bidirectional(CuDNNLSTM((data_dim), return_sequences=True), batch_input_shape=(None, timesteps,data_dim)))
        Model.model.add(Dense(160, activation='sigmoid'))
        
        # compile settings
        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
#        print(Model.model.summary())
        return Model.model

    def build_CNN_model(dataSize):
        imageHeight = dataSize[0]
        imageWidth = dataSize[1]
        channels = dataSize[2]
        
        Model.model = Sequential()
        
        # layers
        Model.model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(imageHeight,imageWidth,channels)))
        Model.model.add(layers.MaxPooling2D((2,2)))
        Model.model.add(layers.Conv2D(64,(3, 3), activation='relu'))
        Model.model.add(layers.MaxPooling2D((2,2)))
        Model.model.add(layers.Conv2D(128,(3, 3), activation='relu'))
        Model.model.add(layers.MaxPooling2D((2,2)))
        Model.model.add(layers.Flatten())
        Model.model.add(layers.Dense(imageHeight*imageWidth*3, activation='relu'))
        Model.model.add(layers.Dense(imageHeight*imageWidth*5, activation='sigmoid'))
        
        #print(Model.model.summary());
        Model.model.compile(loss='binary_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
        return Model.model
        


    def save_model(*args):
        if len(args) == 0:
            Name = "model"
        elif len(args) == 1:
            Name = args[0]

        model_json = Model.model.to_json()        
        with open("../output/" + Name + ".json", "w") as json_file:
            json_file.write(model_json)
        Model.model.save_weights("../output/" + Name + ".h5")
             
    def load_model(*args):
        if len(args) == 0:
            Name = "model"
        elif len(args) == 1:
            Name = args[0]

        json_file = open("../output/" + Name + ".json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("../output/" + Name + ".h5")
        return loaded_model

    def build_CNN_BI_LSTM_model(dataSize):
        imageHeight = dataSize[0]
        imageWidth = dataSize[1]
        channels = dataSize[2]
        Model.model = Sequential()
        Model.model.add((layers.Conv2D(128,(3,3), activation='relu', input_shape=(imageHeight, imageWidth, channels))))
        Model.model.add((layers.MaxPooling2D((2,2))))
        Model.model.add((layers.Conv2D(64,(3, 3), activation = 'relu')))
        Model.model.add((layers.MaxPooling2D((2,2))))
        Model.model.add((layers.Conv2D(64,(3, 3), activation = 'relu')))
        #Model.model.add(TimeDistributed(layers.Flatten()))
        Model.model.add(layers.Reshape((16,64),name='predictions'))
        # input layer
        Model.model.add(Bidirectional(CuDNNLSTM((64), return_sequences=True)))
        Model.model.add((layers.Flatten()))
        Model.model.add(layers.Dense(imageHeight*imageWidth*5, activation='sigmoid'))
        
        # compile settings
        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        print(Model.model.summary())
        return Model.model

    def ReNet(dataSize):
        imageHeight = dataSize[0]
        imageWidth = dataSize[1]
        channels = dataSize[2]
        input = layers.Input(shape=(imageHeight,imageWidth,channels))
        Timestep = int(imageHeight*imageWidth*0.5*0.5);
        reshapedinput = layers.Reshape((Timestep,2*2*channels),name='inputTimedistribution')(input)        
#        xUp = layers.Dense(32*32*3, activation = 'relu')(input)
#        xDown = layers.Dense(32*32*3,activation = 'relu')(input)
        lstmoutput = 131072
        xUp =   layers.LSTM((256), return_sequences=True)(reshapedinput)
        xDown = layers.LSTM((256),go_backwards = True, return_sequences=True)(reshapedinput)
        xUp = layers.Reshape((16,16,256),name='')(xUp)
        xDown = layers.Reshape((16,16,256),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        out = layers.Dense(4)(concatenate)
        Model.model =  keras.models.Model(inputs=input,outputs=out)

        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        print(Model.model.summary())
        return Model.model