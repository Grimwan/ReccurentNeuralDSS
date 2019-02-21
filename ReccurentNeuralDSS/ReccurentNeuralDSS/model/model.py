import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
from keras import optimizers


class Model:
    """Utility class to represent a model."""
    
    model = 0
        
    def build_model(dataSize):
        Model.model = Sequential()
        
        # input layer
        Model.model.add(CuDNNLSTM(128, input_shape=(dataSize, ), return_sequences=True))
        Model.model.add(Dropout(0.2))
        
        # hidden layers
        Model.model.add(CuDNNLSTM(128))
        Model.model.add(Dropout(0.1))
        
        Model.model.add(Dense(32, activation='relu'))
        Model.model.add(Dropout(0.2))
        
        # output layer
        Model.model.add(Dense(dataSize, activation='softmax'))
        
        opt = optimizers.Adam(lr=0.001, decay=1e-6)
        
        # compile settings
        Model.model.compile(loss='binarized_crossentropy', 
                      optimizer=opt, 
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

