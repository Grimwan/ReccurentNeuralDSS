import matplotlib.pyplot as plt
import utils.imageLoader as loader
from model.model import Model

DATADIR = "../Imagesfiles"
#Training = ["CB55/img/training", "CS18/img/training","CS863/img/training"]
#Result = ["CB55/pixel-level-gt/training", "CS18/pixel-level-gt/training", "CS863/pixel-level-gt/training"]
Training = ["DeansTestmap/img/training"]
Result = ["DeansTestmap/pixel-level-gt/training"]
Xsize = 32
Ysize= 32

def main():
    loader.ImageLoader.saveImagesToPickle(DATADIR, Training, Result, [0,2000,2000], [1,Xsize,Ysize], True, "X.pickle", "y.pickle", "../output/");
    [x_train,y_train] = loader.ImageLoader.loadSavedImage("../output/", "X.pickle", "y.pickle")
    
    print(x_train.shape)
    
    x_train = x_train.reshape(y_train.shape[0], Xsize*Ysize*3)
    x_train = x_train.astype('float32') / 255
    y_train = y_train.reshape(y_train.shape[0], Xsize*Ysize*3)
    y_train = y_train.astype('float32') / 255
    
    loader.ImageLoader.removingOnlyDarkpictures(y_train,0)
    
    model = Model.build_model(Xsize*Ysize*3)
    model.fit(x_train,y_train, epochs=1, batch_size=20)
    ynew = model.predict(x_train)
    
    
    x_train = x_train.reshape(x_train.shape[0],Xsize,Ysize,3)
    x_train = x_train.astype('float32') * 255
    y_train = y_train.reshape(y_train.shape[0],Xsize,Ysize,3)
    y_train = y_train.astype('float32') * 255
    
    ynew = ynew.reshape(ynew.shape[0],Xsize,Ysize,3)
    ynew = ynew.astype('float32') * 255
    
    img = loader.ImageLoader.combine_images(ynew, 6496, 4872)
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

