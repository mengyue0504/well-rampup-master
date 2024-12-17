@echo off
REM Log the start of the process
echo Starting script execution at %date% %time%
echo ----------------------------------------

REM Run the first script
echo Running C:\ShenziRampup\WebID.py...
python C:\ShenziRampup\WebID.py
if %errorlevel% neq 0 (
    echo WebID.py failed with error level %errorlevel%. Exiting...
    exit /b %errorlevel%
)

REM Run the second script
echo Running C:\ShenziRampup\runStandingInstructionsParserPIWebAPI.py...
python C:\ShenziRampup\runStandingInstructionsParserPIWebAPI.py
if %errorlevel% neq 0 (
    echo runStandingInstructionsParserPIWebAPI.py failed with error level %errorlevel%. Exiting...
    exit /b %errorlevel%
)

REM Log successful completion
echo ----------------------------------------
echo All scripts completed successfully at %date% %time%.
exit /b 0
