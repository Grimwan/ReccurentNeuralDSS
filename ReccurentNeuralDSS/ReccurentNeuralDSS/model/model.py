import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, CuDNNLSTM, Bidirectional, TimeDistributed
from keras import layers
import keras
import keras.backend as K
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
        input = layers.Input(shape=(imageHeight,imageWidth,channels))
        layer =layers.Conv2D(128,(3,3), activation='relu')(input)
        layer =layers.MaxPooling2D((2,2))(layer)
        layer =layers.Conv2D(64,(3,3), activation='relu')(layer)
        layer =layers.MaxPooling2D((2,2))(layer)
        Split =layers.Conv2D(64,(3,3), activation='relu')(layer)
        
        #firstreadUpdown        
        Split1 = layers.Reshape((Split._keras_shape[1]*Split._keras_shape[2],Split._keras_shape[3]),name='AddingTimeStepsupdown')(Split)
        Split1 = Bidirectional(CuDNNLSTM((Split1._keras_shape[2]), return_sequences=True))(Split1)

        #second read from side to side
        Split2 = layers.Lambda(Model.rotateMatrix)(Split)
        Split2 = layers.Reshape((Split2._keras_shape[1]*Split2._keras_shape[2],Split2._keras_shape[3]),name='AddingTimeStepsleftright')(Split2)
        Split2 = Bidirectional(CuDNNLSTM((Split2._keras_shape[2]), return_sequences=True))(Split2)

        layer = layers.concatenate(inputs = [Split1,Split2],axis=-1)

        layer = ((layers.Flatten()))(layer)
        out = (layers.Dense(imageHeight*imageWidth*5, activation='sigmoid'))(layer)
        Model.model =  keras.models.Model(inputs=input,outputs=out)


        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        print(Model.model.summary())

        return Model.model

    def HorisontalySweepLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        input = args[3]
        Timestep = int(imageHeight * imageWidth*2);
        input = layers.Lambda(Model.rotateMatrix)(input)
        reshapedinput = layers.Reshape((Timestep,channels),name='')(input)
        xUp =   layers.CuDNNLSTM(int(channels/2), return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM(int(channels/2),go_backwards = True, return_sequences=True)(reshapedinput)
        xUp =   layers.Reshape((int(imageHeight),int(imageWidth),channels),name='')(xUp)
        xDown = layers.Reshape((int(imageHeight),int(imageWidth),channels),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        concatenate = layers.Lambda(Model.rotateMatrix)(concatenate)
        channels = channels
        return [concatenate,imageHeight,imageWidth,channels]

    def VerticalysweepLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        input = args[3]
        Timestep = int(imageHeight * imageWidth*0.5);
        reshapedinput = layers.Reshape((Timestep,channels*4),name='')(input)
        xUp =   layers.CuDNNLSTM(int(channels/2), return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM(int(channels/2),go_backwards = True, return_sequences=True)(reshapedinput)
        xUp =   layers.Reshape((int(imageHeight*0.5),int(imageWidth*0.5),channels),name='')(xUp)
        xDown = layers.Reshape((int(imageHeight*0.5),int(imageWidth*0.5),channels),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        channels = channels
        imageHeight = imageHeight *0.5
        imageWidth = imageWidth *0.5
        return [concatenate,imageHeight,imageWidth,channels]

    def FirstVerticalysweepLayer(*args):
        imageHeight = args[0] * 0.5
        imageWidth = args[1] * 0.5
        channels = args[2]
        input = args[3]
        Timestep = int(imageHeight * imageWidth);
        #256,12
        reshapedinput = layers.Reshape((Timestep,channels*4),name='')(input)
        #reshapedinput = keras.layers.transpose_shape(reshapedinput,'channels_first',spatial_axes=(0,3))
        xUp =   layers.CuDNNLSTM((Timestep), return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM((Timestep),go_backwards = True, return_sequences=True)(reshapedinput)
        xUp = layers.Reshape((int(imageHeight),int(imageWidth),Timestep),name='')(xUp)
        xDown = layers.Reshape((int(imageHeight),int(imageWidth),Timestep),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        channels = Timestep
        return [concatenate,imageHeight,imageWidth,channels]




    def reaRangeMatrix(x):
        Whatisthis = K.all(x)
        return Whatisthis
    def rotateMatrix(x):
        if(x.shape.ndims==4):
            transposedValues = K.permute_dimensions(x, (0,2,1,3))
        elif(x.shape.ndims ==3):
            transposedValues = K.permute_dimensions(x, (0,2,1))
        return transposedValues

    def ReNetMiddleLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        concatenate = args[3]
        [concatenate,imageHeight,imageWidth,channels]=Model.VerticalysweepLayer(imageHeight,imageWidth,channels,concatenate)
        [concatenate,imageHeight,imageWidth,channels]=Model.HorisontalySweepLayer(imageHeight,imageWidth,channels,concatenate)
        return [concatenate,imageHeight,imageWidth,channels]

    def ReNet(dataSize):
        imageHeight = dataSize[0]
        imageWidth = dataSize[1]
        channels = dataSize[2]
        input = layers.Input(shape=(imageHeight,imageWidth,channels))
        [concatenate,imageHeight,imageWidth,channels]=Model.FirstVerticalysweepLayer(imageHeight,imageWidth,channels,input)
        [concatenate,imageHeight,imageWidth,channels]=Model.HorisontalySweepLayer(imageHeight,imageWidth,channels,concatenate)
        #Enter amount of ReNetLayers from here on
        [concatenate,imageHeight,imageWidth,channels]=Model.ReNetMiddleLayer(imageHeight,imageWidth,channels,concatenate)
        [concatenate,imageHeight,imageWidth,channels]=Model.ReNetMiddleLayer(imageHeight,imageWidth,channels,concatenate)
        #end of amount of ReNetLayers
        out = layers.Flatten()(concatenate)
        out = (layers.Dense(int(32*32*5), activation='sigmoid'))(out)
        Model.model =  keras.models.Model(inputs=input,outputs=out)

        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        print(Model.model.summary())
        return Model.model