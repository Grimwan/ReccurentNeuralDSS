#### coding rules ####
#liten bokstav på funktioner
#liten bokstav på variabler
import matplotlib.pyplot as plt
import ImageLoader as Loader
DATADIR = "../Imagesfiles"
#Training = ["CB55/img/training", "CS18/img/training","CS863/img/training"]
#Result = ["CB55/pixel-level-gt/training", "CS18/pixel-level-gt/training", "CS863/pixel-level-gt/training"]
Training = ["DeansTestmap/img/training"]
Result = ["DeansTestmap/pixel-level-gt/training"]
Loader.ImageLoader.saveImages(DATADIR,Training,Result,[0,2000,2000],[1,128,128],True,"Testsavehere.x","TestSaveHere.y","../PickleSave/");
[x_train,y_train] = Loader.ImageLoader.loadSavedImage("../PickleSave/","Testsavehere.x","TestSaveHere.y")


#im2 = x_train[42].copy()
#im2[:, :, 0] = x_train[42][:, :, 2]
#im2[:, :, 2] = x_train[42][:, :, 0]
#x_train[42] = im2
max_y,max_x = x_train[0].shape[:2]
print(max_y)
print(max_x)
plt.imshow(x_train[0])
plt.show()

