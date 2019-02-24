import matplotlib.pyplot as plt
from utils.imageLoader import ImageLoader
from model.model import Model
import utils.config as conf
from keras import preprocessing
def main():
    x_train = ImageLoader.load_from_pickle(conf.Picklefiles, "img.pickle")
    y_train = ImageLoader.load_from_pickle(conf.Picklefiles, "gt.pickle")
    #onlyRNN
#    xreshapeValue = [3,conf.Xsize*conf.Ysize*1] #for LSTMRNN
#    yreshapeValue = [3,conf.Xsize*conf.Ysize*1] #for LSTMRNN
    #onlyCNN
#    xreshapeValue = [3,conf.Xsize*conf.Ysize*1] #for LSTMRNN
#    yreshapeValue = [3,conf.Xsize*conf.Ysize*1] #for LSTMRNN



    # flatten
    #x_train, y_train = ImageLoader.flatten_data_multi(x_train, y_train)
    # prepare the model and train it
#    x_train = x_train.reshape(x_train.shape[0], xreshapeValue[0],xreshapeValue[1])
    x_train = x_train.astype('float32') / 255
#    y_train = y_train.reshape(y_train.shape[0],yreshapeValue[0],yreshapeValue[1])
    y_train = ImageLoader.flatten_data(y_train)
#    y_train = y_train.astype('float32') / 255
    print(x_train.shape)
    print(x_train[0].shape)
    #model = Model.build_model(x_train[0].shape)
    #model = Model.build_CNN_robin_model(x_train[0].shape)
    model = Model.build_CNN_model(x_train[0].shape)
    model.fit(x_train, y_train, epochs=10, batch_size=20,validation_split=0.2)
    #Model.save_model()
    # convert to original dimensions
    x_train, y_train = ImageLoader.convert_to_multidimensional_data_multi(x_train, y_train)

    # validation with the original image
    validation = ImageLoader.load_from_pickle(conf.Picklefiles, "combined.pickle")
    #validation = ImageLoader.flatten_data(validation)
#    validation = validation.reshape(validation.shape[0], xreshapeValue[0],xreshapeValue[1])
    validation = validation.astype('float32') / 255
    prediction = model.predict(validation)
    print(prediction.shape)
    prediction = ImageLoader.convert_to_multidimensional_data(prediction)
    
    # show result
    img = ImageLoader.combine_images(prediction, 6496, 4872)
    plt.imshow(img)
    plt.show()
    
    #im2 = x_train[42].copy()
    #im2[:, :, 0] = x_train[42][:, :, 2]
    #im2[:, :, 2] = x_train[42][:, :, 0]
    #x_train[42] = im2
    #max_y,max_x = x_train[0].shape[:2]
    #print(max_y)
    #print(max_x)
    #print(x_train.shape)
    #plt.imshow(img)
    #plt.show()

if __name__ == "__main__":
    main()
