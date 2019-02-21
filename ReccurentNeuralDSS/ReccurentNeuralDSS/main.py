import matplotlib.pyplot as plt
from utils.imageLoader import ImageLoader
from model.model import Model
import utils.config as conf

def main():
    x_train = ImageLoader.load_from_pickle(conf.Picklefiles, "img.pickle")
    y_train = ImageLoader.load_from_pickle(conf.Picklefiles, "gt.pickle")

    # flatten
    x_train, y_train = ImageLoader.flatten_data_multi(x_train, y_train)
    
    # prepare the model and train it
    model = Model.build_model(conf.Xsize*conf.Ysize*3)
    model.fit(x_train, y_train, epochs=1, batch_size=20)

    # convert to original dimensions
    x_train, y_train = ImageLoader.convert_to_multidimensional_data_multi(x_train, y_train)

    # validation with the original image
    validation = ImageLoader.load_from_pickle(conf.Picklefiles, "combined.pickle")
    validation = ImageLoader.flatten_data(validation)

    prediction = model.predict(validation)
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
