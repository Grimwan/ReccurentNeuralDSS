import tensorflow as tf


class ModelClass(object):
    Model = 0
    def build_Standard_NN_model(input,output):
        model = tf.keras.models.Sequential() # feed forward one of two
        #model.add(tf.keras.layers.Flatten()) # flattens a mutli array in to single array from 3d to 1d basically
        #First hidden layer
        model.add(tf.keras.layers.Dense(200, activation = tf.nn.relu, input_shape=(input,))) # tf.nn.relu = a kind of activation like sigmoid function
        #second hidden layer 
        model.add(tf.keras.layers.Dense(200, activation = tf.nn.relu))
        model.add(tf.keras.layers.Dense(200, activation = tf.nn.relu))
        #output layer of neural network
        model.add(tf.keras.layers.Dense(output, activation = tf.nn.sigmoid)) 
        #nn done set up 

        #setting up how to train the neural network specifiyng optimiser and loss function
        model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy']) #lossfunction the degree of error minimise loss, optimizer is the backprogation type. keras supports alot, adam is one of them. for loss the most popular one is categorical crossentropy  
        Model = model
        return model
