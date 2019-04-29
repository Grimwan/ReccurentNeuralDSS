@echo off
cd gtImages
ren *.png gt_image.*
cd ..
cd predictionImages
ren *.png prediction_image.*
cd ..

java -jar LayoutAnalysisEvaluator.jar -gt gtImages/gt_image.png -p predictionImages/prediction_image.png > Output.txt