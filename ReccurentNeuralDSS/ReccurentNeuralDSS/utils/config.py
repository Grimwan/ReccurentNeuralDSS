# directories
DATADIR = "../Imagesfiles"
#Training = ["CB55/img/training"]
#Result = ["CB55/pixel-level-gt/training"]
TestTraining = ["DeansTestmap/img/training"]
TestResult = ["DeansTestmap/pixel-level-gt/training"]

Training = ["CS18/img/training"]
Result = ["CS18/pixel-level-gt/training"]
ValidationTraining = ["CS18/img/validation"]
ValidationResult = ["CS18/pixel-level-gt/validation"]
PredictionPictureTraining= ["CS18/img/public-test"]
PredictionPictureResult= ["CS18/pixel-level-gt/public-test"]
WhereTosaveTestImage = "here"
NameOfTestImage = "testimage"

# variables
Xsize = 32
Ysize= 32
AmountOfEpochs = 1
batchSize = 50
validationSplit = 0.2
orignalPictureX = 4992
orignalPictureY = 3328