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
        
        # input layer
        Model.model.add(CuDNNLSTM(1, batch_input_shape=(None,3072,1), return_sequences=True))
        #Model.model.add(LSTM((1),batch_input_shape=(None,3072,1),return_sequences=True))
        # compile settings
        Model.model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
        
        return Model.model

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

