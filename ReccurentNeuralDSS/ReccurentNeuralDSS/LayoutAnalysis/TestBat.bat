@echo off
setlocal EnableDelayedExpansion
set i=0
for %%a in (predictionImages/CB55/*) do (
   set listCB55[!i!]=%%a
   set /A i+=1
)
set i=0
for %%a in (predictionImages/CS18/*) do (
   set listCS18[!i!]=%%a
   set /A i+=1
   echo %%a
)
set i=0
for %%a in (predictionImages/cs863/*) do (
   set listCS863[!i!]=%%a
   set /A i+=1
)



set i=0
for %%a in (../../Imagesfiles/CB55/pixel-level-gt/public-test/*) do (
   set gtImageCB55[!i!]=%%a
   set /A i+=1
)

set i=0
for %%a in (../../Imagesfiles/CS18/pixel-level-gt/public-test/*) do (
   set gtImageCS18[!i!]=%%a
   set /A i+=1
   echo %%a
)

set i=0
for %%a in (../../Imagesfiles/CS863/pixel-level-gt/public-test/*) do (
   set gtImageCS863[!i!]=%%a
   set /A i+=1
)

REM echo %i%

REM echo %listCB55[0]%
REM echo %gtImageCB55[1]%
REM echo %gtImageCB55[2]%
set i=0
call set me=%%listCB55[!i!]%%
REM echo %me%



echo CB55 Predictions>Output.txt
for %%a in (../../Imagesfiles/CB55/pixel-level-gt/public-test/*) do (
	call set PredictionImage=%%listCB55[!i!]%%
	call set gtImage=%%gtImageCB55[!i!]%%
 	java -jar LayoutAnalysisEvaluator.jar -gt ../../Imagesfiles/CB55/pixel-level-gt/public-test/!gtImage! -p predictionImages/CB55/!PredictionImage! --outputPath ../../visualisation >> Output.txt  
 	set /A i+=1
)
set i=0
call set PredictionImage=%%listCS18[!i!]%%
call set gtImage=%%gtImageCS18[!i!]%%


echo CS18 >> Output.txt
for %%a in (../../Imagesfiles/CS18/pixel-level-gt/public-test/*) do (
	call set PredictionImage=%%listCS18[!i!]%%
	call set gtImage=%%gtImageCS18[!i!]%%
	java -jar LayoutAnalysisEvaluator.jar -gt ../../Imagesfiles/CS18/pixel-level-gt/public-test/!gtImage! -p predictionImages/CS18/!PredictionImage! --outputPath ../../visualisation >> Output.txt  
	set /A i+=1
)

set i=0
echo CS863 >> Output.txt
for %%a in (../../Imagesfiles/CS863/pixel-level-gt/public-test/*) do (
call set PredictionImage=%%listCS863[!i!]%%
call set gtImage=%%gtImageCS863[!i!]%%
java -jar LayoutAnalysisEvaluator.jar -gt ../../Imagesfiles/CS863/pixel-level-gt/public-test/!gtImage! -p predictionImages/CS863/!PredictionImage! --outputPath ../../visualisation >> Output.txt  
set /A i+=1
)




REM for %%a in (../../Imagesfiles/CB55/pixel-level-gt/public-test/*) do (
 
REM java -jar LayoutAnalysisEvaluator.jar -gt ../../Imagesfiles/CB55/pixel-level-gt/public-test/gtImage -p predictionImages/CB55/PredictionImage --outputPath ../../visualisation >> Output.txt  
REM	set /A i+=1
REM )
REM java -jar LayoutAnalysisEvaluator.jar -gt ../../Imagesfiles/CB55/pixel-level-gt/public-test/%gtImageCB55[0]% -p predictionImages/CB55/%listCB55[0]% --outputPath ../../visualisation >> Output.txt

pause