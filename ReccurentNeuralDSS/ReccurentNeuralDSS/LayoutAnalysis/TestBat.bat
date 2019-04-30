@echo off
setlocal EnableDelayedExpansion
set i=0
for %%a in (gtImages/*) do (
   set /A i+=1
   set list[!i!]=%%a
)
echo %i%
echo %list[0]%
echo %list[1]%
echo %list[2]%

pause