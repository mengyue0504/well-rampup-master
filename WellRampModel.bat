@echo off
REM Run the first Python script (WellRampModel.py)
echo Running WellRampModel.py...
python "C:\ShenziRampup\WellRampModel.py"
if %errorlevel% neq 0 (
    echo WellRampModel.py failed. Exiting...
    exit /b %errorlevel%
)
