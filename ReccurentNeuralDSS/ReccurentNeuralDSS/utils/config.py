# directories
DATADIR = "../Imagesfiles"
#Training = ["CB55/img/training"]
#Result = ["CB55/pixel-level-gt/training"]
TestTraining = ["DeansTestmap/img/training"]
TestResult = ["DeansTestmap/pixel-level-gt/training"]

Training = ["CS18/img/training","CB55/img/training","CS863/img/training"]
Result = ["CS18/pixel-level-gt/training","CB55/pixel-level-gt/training","CS863/pixel-level-gt/training"]
ValidationTraining = ["CS18/img/validation"]
ValidationResult = ["CS18/pixel-level-gt/validation"]
PredictionPictureCB55= ["CB55/img/public-test"]
PredictionPictureResultCB55= ["CB55/pixel-level-gt/public-test"]
PredictionPictureCS= ["CS18/img/public-test","CS863/img/public-test"]
PredictionPictureResultCS= ["CS18/pixel-level-gt/public-test","CS863/pixel-level-gt/public-test"]

WhereTosaveTestImage = "here"
NameOfTestImage = "testimage"

# variables
Xsize = 64
Ysize= 64
AmountOfEpochs = 1
batchSize = 50
validationSplit = 0.2
orignalPictureX = 4992
orignalPictureY = 3328