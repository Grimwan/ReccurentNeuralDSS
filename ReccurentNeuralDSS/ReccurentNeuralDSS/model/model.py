import tensorflow as tf
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Flatten


class Model:
    """Utility class to represent a model."""
    
    model = 0
        
    def build_model(dataSize):
        Model.model = Sequential()
        #model.add(Flatten()) # flattens array to 1D
        
        # hidden layers
        Model.model.add(Dense(200, activation='relu', input_shape=(dataSize,)))
        Model.model.add(Dense(200, activation='relu'))
        Model.model.add(Dense(200, activation='relu'))
        
        # output layer
        Model.model.add(Dense(dataSize, activation='sigmoid')) 

        # compile settings
        Model.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
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

