DATADIR = "../Imagesfiles"
#Training = ["CB55/img/training"]
#Result = ["CB55/pixel-level-gt/training"]
TestTraining = ["DeansTestmap/img/training"]
TestResult = ["DeansTestmap/pixel-level-gt/training"]
Training = ["DeansTestmap/img/training"]
Result = ["DeansTestmap/pixel-level-gt/training"]
ValidationTraining = ["CS18/img/validation"]
ValidationResult = ["CS18/pixel-level-gt/validation"]
Xsize = 32
Ysize= 32
AmountOfEpochs = 1
batchSize = 20
validationSplit = 0.2
Picklefiles = "../output"


LoadTestPickle = "combined.pickle"
WhereTosaveTestImage = "here"
NameOfTestImage = "testimage"

orignalPictureX = 6496
orignalPictureY = 4872