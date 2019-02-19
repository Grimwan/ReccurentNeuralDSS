#### coding rules ####
#liten bokstav på funktioner
#liten bokstav på variabler
import matplotlib.pyplot as plt
import ImageLoader as Loader
import model as model
DATADIR = "../Imagesfiles"
#Training = ["CB55/img/training", "CS18/img/training","CS863/img/training"]
#Result = ["CB55/pixel-level-gt/training", "CS18/pixel-level-gt/training", "CS863/pixel-level-gt/training"]
Training = ["DeansTestmap/img/training"]
Result = ["DeansTestmap/pixel-level-gt/training"]
Xsize = 32
Ysize= 32


Loader.ImageLoader.saveImages(DATADIR,Training,Result,[0,2000,2000],[1,Xsize,Ysize],True,"Testsavehere.x","TestSaveHere.y","../PickleSave/");
[x_train,y_train] = Loader.ImageLoader.loadSavedImage("../PickleSave/","Testsavehere.x","TestSaveHere.y")

print(x_train.shape)

x_train = x_train.reshape(y_train.shape[0],Xsize*Ysize*3)
x_train = x_train.astype('float32') / 255
y_train = y_train.reshape(y_train.shape[0],Xsize*Ysize*3)
y_train = y_train.astype('float32') / 255



Loader.ImageLoader.removingOnlyDarkpictures(y_train,0)

Model = model.ModelClass.build_Standard_NN_model(Xsize*Ysize*3,Xsize*Ysize*3)
Model.fit(x_train,y_train, epochs = 20,batch_size=20)
ynew = Model.predict(x_train)




x_train = x_train.reshape(x_train.shape[0],Xsize,Ysize,3)
x_train = x_train.astype('float32') * 255
y_train = y_train.reshape(y_train.shape[0],Xsize,Ysize,3)
y_train = y_train.astype('float32') * 255

ynew = ynew.reshape(ynew.shape[0],Xsize,Ysize,3)
ynew = ynew.astype('float32') * 255

img=Loader.ImageLoader.combine_imgs(ynew,6496,4872)
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

