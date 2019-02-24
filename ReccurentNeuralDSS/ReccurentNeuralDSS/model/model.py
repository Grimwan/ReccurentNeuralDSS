import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras import optimizers
from keras.layers import Flatten, Dense

class Model:
    """Utility class to represent a model."""
    
    model = 0
       
    def build_model(dataSize):
        Model.model = Sequential()
        batch_size = None
        timesteps = dataSize[0]
        data_dim = dataSize[1]
        # input layer
        Model.model.add(CuDNNLSTM((data_dim), batch_input_shape=(None,timesteps,data_dim), return_sequences=True))
        Model.model.add(Dense(data_dim,activation='sigmoid'))
        #Model.model.add(LSTM((data_dim), batch_input_shape=(None,timesteps,data_dim), activation = 'sigmoid', return_sequences=True))
        # compile settings
        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        
        return Model.model

#    def build_model_CNNLSTM(input_Shape):
    #def build_CNN_model()    
        
    def save_model():
        model_json = Model.model.to_json()
        
        with open("../output/model.json", "w") as json_file:
            json_file.write(model_json)
        Model.model.save_weights("../output/model.h5")
        
    def load_model():
        json_file = open('../ouput/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("..output/model.h5")
        return loaded_model

#lstm comment: For example, say your input sequences look like X = [[0.54, 0.3], [0.11, 0.2], [0.37, 0.81]]. We can see that this sequence has a timestep of 3 and a data_dim of 2.