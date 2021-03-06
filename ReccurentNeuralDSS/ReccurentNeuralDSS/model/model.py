import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, CuDNNLSTM, Bidirectional, TimeDistributed
from keras import layers
import keras
import keras.backend as K

#############UNET#Import###############
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.core import SpatialDropout2D, Activation
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
#######################################
class Unet:

    model = 0
    def double_conv_layer(inputs, filter):
        conv = Conv2D(filter, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
        conv = BatchNormalization(axis=3)(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(filter, (3, 3), padding='same', kernel_initializer='he_normal')(conv)
        conv = BatchNormalization(axis=3)(conv)
        conv = Activation('relu')(conv)
        conv = SpatialDropout2D(0.1)(conv)
        return conv

    def down_layer(inputs, filter):
        """Create downsampling layer."""
        conv = Unet.double_conv_layer(inputs, filter)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    def up_layer(inputs, concats, filter):
        """Create upsampling layer."""
        return Unet.double_conv_layer(concatenate([UpSampling2D(size=(2, 2))(inputs), concats], axis=3), filter)


    def BuildUnet(dataSize):
        imageHeight = dataSize[0]
        imageWidth = dataSize[1]
        channels = dataSize[2]
        """Create U-net."""
        inputs = Input((imageHeight, imageWidth, channels))

        # Downsampling.
        down1, pool1 = Unet.down_layer(inputs, 32)
        down2, pool2 = Unet.down_layer(pool1, 64)
        down3, pool3 = Unet.down_layer(pool2, 128)
        down4, pool4 = Unet.down_layer(pool3, 256)
        down5, pool5 = Unet.down_layer(pool4, 512)

        # Bottleneck.
        bottleneck = Unet.double_conv_layer(pool5, 1024)

        # Upsampling.
        up5 = Unet.up_layer(bottleneck, down5, 512)
        up4 = Unet.up_layer(up5, down4, 256)
        up3 = Unet.up_layer(up4, down3, 128)
        up2 = Unet.up_layer(up3, down2, 64)
        up1 = Unet.up_layer(up2, down1, 32)

        outputs = Conv2D(5, (1, 1))(up1)
        outputs = Activation('sigmoid')(outputs)
        outputs = layers.Flatten()(outputs)
        Unet.model = keras.models.Model(inputs, outputs)
        return Unet.model



class ReDeaNet:
    model = 0

    def reaRangeMatrix(x):
        Whatisthis = K.all(x)
        return Whatisthis
    def rotateMatrix(x):
        if(x.shape.ndims==4):
            transposedValues = K.permute_dimensions(x, (0,2,1,3))
        elif(x.shape.ndims ==3):
            transposedValues = K.permute_dimensions(x, (0,2,1))
        return transposedValues
    def HorisontalySweepLayer(input,filter):
        height = input._keras_shape[1]
        width = input._keras_shape[2]
        Channels = input._keras_shape[3]
        Timestep = int(width * height);
        input = layers.Lambda(Model.rotateMatrix)(input)
        reshapedinput = layers.Reshape((int(Timestep),int(Channels)),name='')(input)
        xUp =   layers.CuDNNLSTM(int(filter/2),unit_forget_bias=True, return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM(int(filter/2),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
        xUp =   layers.Reshape((int(height),int(width),int(filter/2)),name='')(xUp)
        xDown = layers.Reshape((int(height),int(width),int(filter/2)),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        concatenate = layers.Lambda(Model.rotateMatrix)(concatenate)
        return concatenate
    def verticalSweepLayer(input,filter,Scale,Down):
        height = input._keras_shape[1]
        width = input._keras_shape[2]
        Channels = input._keras_shape[3]
        Scale = Scale * 2
        Timestep = int(width * height*(1/Scale));
        reshapedinput = layers.Reshape((int(Timestep),int(Channels*Scale)),name='')(input)
        if(Down):
            xUp =   layers.CuDNNLSTM(int(filter/2),unit_forget_bias=True, return_sequences=True)(reshapedinput)
            xDown = layers.CuDNNLSTM(int(filter/2),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
            xUp =   layers.Reshape((int(height*(2/Scale)),int(width*(2/Scale)),int(filter/2)),name='')(xUp)
            xDown = layers.Reshape((int(height*(2/Scale)),int(width*(2/Scale)),int(filter/2)),name='')(xDown)
        else:
            xUp =   layers.CuDNNLSTM(int(filter*Scale*2),unit_forget_bias=True, return_sequences=True)(reshapedinput)
            xDown = layers.CuDNNLSTM(int(filter*Scale*2),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
            xUp =   layers.Reshape((int(height*(Scale/2)),int(width*(Scale/2)),int(filter/2)),name='')(xUp)
            xDown = layers.Reshape((int(height*(Scale/2)),int(width*(Scale/2)),int(filter/2)),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        return concatenate
    def down_layer(input,filter,Scale):
        input = ReDeaNet.verticalSweepLayer(input,filter,Scale,True)
        input = ReDeaNet.HorisontalySweepLayer(input,filter)
        return input
    def up_layer(input,filter,Scale):
        input = ReDeaNet.verticalSweepLayer(input,filter,Scale,False)
        input = ReDeaNet.HorisontalySweepLayer(input,filter)
        return input
    def BuildDeaNet(dataSize):
        imageHeight = dataSize[0]
        imageWidth = dataSize[1]
        channels = dataSize[2]
        """Create ReDeaNet."""
        inputs = Input((imageHeight, imageWidth, channels))
        down1 = ReDeaNet.down_layer(inputs, 4,2)
        down2 = ReDeaNet.down_layer(down1, 64,2)
        up1 = ReDeaNet.up_layer(down2, 64,2)
        up2 = ReDeaNet.up_layer(up1, 4,2)
        outputs = layers.Flatten()(up2)
        Unet.model = keras.models.Model(inputs, outputs)
        return Unet.model


#lstm: For example, say your input sequences look like X = [[0.54, 0.3], [0.11, 0.2], [0.37, 0.81]]. We can see that this sequence has a timestep of 3 and a data_dim of 2.
class Model:
    """Utility class to represent a model."""
    
    model = 0
       
    def build_BI_LSTM_model(dataSize):
        Model.model = Sequential()
        timesteps = dataSize[0] # first layer array size
        data_dim = dataSize[1] # how big is one part in the array.
        

        input = layers.Input(shape=(timesteps,data_dim))       
        #firstreadUpdown          
        Split1 = Bidirectional(CuDNNLSTM((data_dim), unit_forget_bias=True,return_sequences=True))(input)
        #second read from side to side
        Split2 = layers.Lambda(Model.rotateMatrix)(input)
        Split2 = layers.Reshape((timesteps,data_dim),name='ReshapeingforTimesteps')(Split2)
        Split2 = Bidirectional(CuDNNLSTM((data_dim),unit_forget_bias=True, return_sequences=True))(Split2)

        layer = layers.concatenate(inputs = [Split1,Split2],axis=-1)


        layer = ((layers.Flatten()))(layer)
        out = (layers.Dense((int)(timesteps*(data_dim/3)*5), activation='sigmoid'))(layer)
        Model.model =  keras.models.Model(inputs=input,outputs=out)

        # input layer
        #Model.model.add(Bidirectional(CuDNNLSTM((data_dim), return_sequences=True), batch_input_shape=(None, timesteps,data_dim)))
        #Model.model.add(Dense(160, activation='sigmoid'))
        
        # compile settings
        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        print(Model.model.summary())
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
        
    def build_Unet_model(dataSize):
        #locked to 128 ;)
        Model.model = Unet.BuildUnet(dataSize)
        print(Model.model.summary())
        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        return Model.model
    def build_ReDeaNet_model(dataSize):
        Model.model = ReDeaNet.BuildDeaNet(dataSize)
        print(Model.model.summary())
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
        Split1 = Bidirectional(CuDNNLSTM((Split1._keras_shape[2]),unit_forget_bias=True, return_sequences=True))(Split1)

        #second read from side to side
        Split2 = layers.Lambda(Model.rotateMatrix)(Split)
        Split2 = layers.Reshape((Split2._keras_shape[1]*Split2._keras_shape[2],Split2._keras_shape[3]),name='AddingTimeStepsleftright')(Split2)
        Split2 = Bidirectional(CuDNNLSTM((Split2._keras_shape[2]),unit_forget_bias=True, return_sequences=True))(Split2)

        layer = layers.concatenate(inputs = [Split1,Split2],axis=-1)

        layer = ((layers.Flatten()))(layer)
        out = (layers.Dense(imageHeight*imageWidth*5, activation='sigmoid'))(layer)
        Model.model =  keras.models.Model(inputs=input,outputs=out)


        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        print(Model.model.summary())

        return Model.model

    def HorisontalySweepUpLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        input = args[3]
        Timestep = int(imageHeight * imageWidth);

        input = layers.Lambda(Model.rotateMatrix)(input)
        reshapedinput = layers.Reshape((Timestep,channels),name='')(input)
        xUp =   layers.CuDNNLSTM(int(channels/2),unit_forget_bias=True, return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM(int(channels/2),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
        xUp =   layers.Reshape((int(imageHeight),int(imageWidth),int(channels/2)),name='')(xUp)
        xDown = layers.Reshape((int(imageHeight),int(imageWidth),int(channels/2)),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        concatenate = layers.Lambda(Model.rotateMatrix)(concatenate)
        channels = channels/2
        return [concatenate,imageHeight,imageWidth,channels]

    def HorisontalySweepLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        input = args[3]
        Timestep = int(imageHeight * imageWidth*2);
        input = layers.Lambda(Model.rotateMatrix)(input)
        reshapedinput = layers.Reshape((Timestep,channels),name='')(input)
        xUp =   layers.CuDNNLSTM(int(channels/2),unit_forget_bias=True, return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM(int(channels/2),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
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
        xUp =   layers.CuDNNLSTM(int(channels/2),unit_forget_bias=True, return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM(int(channels/2),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
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
        xUp =   layers.CuDNNLSTM((Timestep),unit_forget_bias=True, return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM((Timestep),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
        xUp = layers.Reshape((int(imageHeight),int(imageWidth),Timestep),name='')(xUp)
        xDown = layers.Reshape((int(imageHeight),int(imageWidth),Timestep),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        channels = Timestep
        return [concatenate,imageHeight,imageWidth,channels]


    def VerticalysweepUpscaleLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        input = args[3]
        Timestep = int(imageHeight * imageWidth);
        reshapedinput = layers.Reshape((Timestep,int(channels*2)),name='')(input)
        xUp =   layers.CuDNNLSTM(int(channels),unit_forget_bias=True, return_sequences=True)(reshapedinput)
        xDown = layers.CuDNNLSTM(int(channels),unit_forget_bias=True,go_backwards = True, return_sequences=True)(reshapedinput)
        xUp =   layers.Reshape((int(imageHeight*2),int(imageWidth*2),int(channels/4)),name='')(xUp)
        xDown = layers.Reshape((int(imageHeight*2),int(imageWidth*2),int(channels/4)),name='')(xDown)
        concatenate = layers.concatenate(inputs = [xUp,xDown],axis=-1)
        channels = int(channels/2)
        imageHeight = imageHeight*2
        imageWidth = imageWidth *2
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

    def ReNetScaleDownLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        concatenate = args[3]
        [concatenate,imageHeight,imageWidth,channels]=Model.VerticalysweepLayer(imageHeight,imageWidth,channels,concatenate)
        [concatenate,imageHeight,imageWidth,channels]=Model.HorisontalySweepLayer(imageHeight,imageWidth,channels,concatenate,0)
        return [concatenate,imageHeight,imageWidth,channels]

    def ReNetScaleUpLayer(*args):
        imageHeight = args[0]
        imageWidth = args[1]
        channels = args[2]
        concatenate = args[3]
        [concatenate,imageHeight,imageWidth,channels]=Model.VerticalysweepUpscaleLayer(imageHeight,imageWidth,channels,concatenate)
        [concatenate,imageHeight,imageWidth,channels]=Model.HorisontalySweepUpLayer(imageHeight,imageWidth,channels,concatenate,0)
        return [concatenate,imageHeight,imageWidth,channels]

    def ReNet(dataSize):
        imageHeight = dataSize[0]
        imageWidth = dataSize[1]
        channels = dataSize[2]
        input = layers.Input(shape=(imageHeight,imageWidth,channels))
        [concatenate,imageHeight,imageWidth,channels]=Model.FirstVerticalysweepLayer(imageHeight,imageWidth,channels,input)
        [concatenate,imageHeight,imageWidth,channels]=Model.HorisontalySweepLayer(imageHeight,imageWidth,channels,concatenate)
        #Enter amount of ReNetLayers from here on
        [concatenate,imageHeight,imageWidth,channels]=Model.ReNetScaleDownLayer(imageHeight,imageWidth,channels,concatenate)
        #end of amount of ReNetLayers
        out = layers.Flatten()(concatenate)
        out = (layers.Dense(int(32*32*5), activation='sigmoid'))(out)
        Model.model =  keras.models.Model(inputs=input,outputs=out)

        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        print(Model.model.summary())
        return Model.model