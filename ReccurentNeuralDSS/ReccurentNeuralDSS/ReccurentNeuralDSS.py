#### coding rules ####
#liten bokstav på funktioner
#liten bokstav på variabler
import matplotlib.pyplot as plt
import ImageLoader as Loader
DATADIR = "../Imagesfiles"
Training = ["CB55/img/training", "CS18/img/training","CS863/img/training"]
Result = ["CB55/pixel-level-gt/training", "CS18/pixel-level-gt/training", "CS863/pixel-level-gt/training"]
#Loader.ImageLoader.saveImages(DATADIR,Training,Result,[1,2000,2000],True,"Testsavehere.x","TestSaveHere.y");
[x_train,y_train] = Loader.ImageLoader.loadSavedImage("Testsavehere.x","TestSaveHere.y")

print(x_train.shape[2])
print(y_train.shape)
plt.imshow(x_train[42])
plt.show()

