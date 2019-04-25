import train 
import subprocess
from utils.imageLoader import ImageLoader

def main():
    #train.Train("StandardCNN")
    #train.Train("BiDirectionalLSTMRNN")
    train.PredictImages("somethingelse",False);
    #train.ExperimentTraining()
    #train.Train("ReNet")
    #train.ImageLoaderTest()
    #print(subprocess.check_output(["RunMe.bat"],cwd = "LayoutAnalysis" ,shell= True))
    #ImageLoader.MovePicture("FixImages","testimage0",-12,0)
    #hm = 2
if __name__ == "__main__":
    main()
